import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Moon, Sun, Activity, Database, AlertCircle, 
  TrendingUp, Send, CheckCircle, XCircle 
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import ReactMarkdown from 'react-markdown';

const API_URL = 'http://localhost:8000/v1';

export default function App() {
  const [theme, setTheme] = useState('dark');
  const [logs, setLogs] = useState([]);
  const [driftEvents, setDriftEvents] = useState([]);
  const [health, setHealth] = useState({ game_features: {}, user_features: {} });
  const [recentActions, setRecentActions] = useState([]);
  
  // Form State
  const [formData, setFormData] = useState({
    review_text: '',
    game_name: '',
    user_id: '',
    genre: '',
    recommended: 'Yes'
  });
  const [submitStatus, setSubmitStatus] = useState(null);
  const [showJson, setShowJson] = useState(false);

  useEffect(() => {
    document.documentElement.className = theme;
  }, [theme]);

  const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark');

  const fetchData = async () => {
    try {
      const [logsRes, driftRes, healthRes, actionsRes] = await Promise.all([
        axios.get(`${API_URL}/logs?limit=10`),
        axios.get(`${API_URL}/drift_events?limit=5`),
        axios.get(`${API_URL}/cluster_health`),
        axios.get(`http://localhost:8000/recent_actions`)
      ]);
      setLogs(logsRes.data);
      setDriftEvents(driftRes.data);
      setHealth(healthRes.data);
      setRecentActions(actionsRes.data);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitStatus('loading');
    try {
      const res = await axios.post(`${API_URL}/submit_review`, formData);
      if (res.data.status === 'success') {
        setSubmitStatus({ 
          type: 'success', 
          msg: `Score: ${res.data.predict.intelligent_score.toFixed(3)} | Path: ${res.data.predict.decision_path}`,
          auditLog: res.data.predict.llm_audit_log,
          contract: res.data.result
        });
        setFormData({ review_text: '', game_name: '', user_id: '', genre: '', recommended: 'Yes' });
        setShowJson(false);
        fetchData();
      } else {
        setSubmitStatus({ type: 'error', msg: res.data.message });
      }
    } catch (error) {
      setSubmitStatus({ type: 'error', msg: 'Failed to submit review' });
    }
  };

  // Recharts needs an array of objects for the SHAP trend line
  const shapData = recentActions.slice().reverse().map((action, i) => ({
    name: i,
    cosine: action.shap_cosine
  }));

  const totalLogs = logs.length;
  const passCount = logs.filter(l => l.interaction_metadata?.triage_status === 'Pass').length;
  const passRatio = totalLogs > 0 ? ((passCount / totalLogs) * 100).toFixed(1) : 0;

  return (
    <div style={{ minHeight: '100vh', padding: '2rem' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3rem' }}>
        <div>
          <h1>PixelProspector V4.0</h1>
          <p className="text-muted" style={{ marginTop: '0.5rem' }}>Agentic Flywheel Dashboard</p>
        </div>
        <button onClick={toggleTheme} className="nm-button" aria-label="Toggle Theme">
          {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
        </button>
      </header>

      <div className="grid grid-cols-3">
        {/* Zone 1: Cluster Health */}
        <div className="nm-card">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.25rem' }}>
            <Activity className="text-primary" /> Cluster Health
          </h2>
          <div style={{ marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>GAME FEATURES (AVG)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.875rem' }}>
              {Object.entries(health.game_features).slice(0, 4).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>{k.split('_')[0]}:</span> <strong>{v}</strong>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>USER FEATURES (AVG)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.875rem' }}>
              {Object.entries(health.user_features).slice(0, 4).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>{k.split('_')[0]}:</span> <strong>{v}</strong>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Zone 5: Outcome Tracking */}
        <div className="nm-card">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.25rem' }}>
            <Database className="text-info" /> Outcome Tracking
          </h2>
          <div style={{ textAlign: 'center', margin: '2rem 0' }}>
            <div style={{ fontSize: '3rem', fontWeight: 'bold', color: 'var(--success)' }}>{passRatio}%</div>
            <div className="text-muted">Pass Rate (Recent)</div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
            <div className="nm-badge text-success">Pass: {passCount}</div>
            <div className="nm-badge text-danger">Reject: {totalLogs - passCount}</div>
          </div>
        </div>

        {/* Zone 6: System Alerts */}
        <div className="nm-card">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.25rem' }}>
            <AlertCircle className="text-warning" /> System Alerts
          </h2>
          {driftEvents.length === 0 ? (
            <p className="text-muted">No drift events recorded.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {driftEvents.map(event => (
                <div key={event.id} style={{ padding: '1rem', borderLeft: '4px solid var(--warning)', backgroundColor: 'var(--shadow-light)', borderRadius: '0 var(--radius-sm) var(--radius-sm) 0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <strong>Drift Detected</strong>
                    <span className="text-muted" style={{ fontSize: '0.75rem' }}>{new Date(event.detected_at).toLocaleTimeString()}</span>
                  </div>
                  <div style={{ fontSize: '0.875rem' }}>Gap Trend: {event.gap_svm_trend.toFixed(3)}</div>
                  <div className="nm-badge text-warning" style={{ marginTop: '0.5rem' }}>{event.auto_healed ? 'Healed' : 'Active'}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Zone 4: SHAP Reliability */}
        <div className="nm-card" style={{ gridColumn: 'span 3' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.25rem' }}>
            <TrendingUp className="text-primary" /> SHAP Reliability (Cosine Similarity)
          </h2>
          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer>
              <LineChart data={shapData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--shadow-dark)" />
                <XAxis dataKey="name" stroke="var(--text-muted)" />
                <YAxis domain={[0, 1]} stroke="var(--text-muted)" />
                <Tooltip contentStyle={{ backgroundColor: 'var(--surface)', border: 'none', borderRadius: '8px', boxShadow: 'var(--nm-flat)' }} />
                <Line type="monotone" dataKey="cosine" stroke="var(--primary)" strokeWidth={3} dot={{ r: 4, fill: 'var(--primary)' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Zone 2: Submit Review */}
        <div className="nm-card" style={{ gridColumn: 'span 1' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.25rem' }}>
            <Send className="text-success" /> Submit Review
          </h2>
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <input name="game_name" value={formData.game_name} onChange={handleInputChange} placeholder="Game Name" className="nm-input" required />
            <input name="user_id" value={formData.user_id} onChange={handleInputChange} placeholder="User ID" className="nm-input" required />
            <input name="genre" value={formData.genre} onChange={handleInputChange} placeholder="Genre" className="nm-input" required />
            
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', fontSize: '0.875rem' }}>
              <label>Recommended?</label>
              <select name="recommended" value={formData.recommended} onChange={handleInputChange} className="nm-input" style={{ width: 'auto' }}>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            
            <textarea name="review_text" value={formData.review_text} onChange={handleInputChange} placeholder="Write review here..." className="nm-input" rows="4" required minLength="20" />
            
            <button type="submit" className="nm-button nm-button-primary" disabled={submitStatus === 'loading'}>
              {submitStatus === 'loading' ? 'Analyzing...' : 'Submit to Pipeline'}
            </button>
            
            {submitStatus && submitStatus !== 'loading' && (
              <div style={{ marginTop: '1rem', padding: '1rem', borderRadius: '8px', backgroundColor: submitStatus.type === 'success' ? 'var(--shadow-light)' : 'rgba(255, 33, 87, 0.1)', color: submitStatus.type === 'success' ? 'var(--success)' : 'var(--danger)' }}>
                {submitStatus.type === 'success' ? <CheckCircle size={16} style={{ display: 'inline', marginRight: '0.5rem' }} /> : <XCircle size={16} style={{ display: 'inline', marginRight: '0.5rem' }}/>}
                {submitStatus.msg}
                
                {submitStatus.auditLog && (
                  <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--shadow-dark)', color: 'var(--text)', fontSize: '0.875rem' }}>
                    <strong style={{ display: 'block', marginBottom: '0.5rem' }}>LLM Generation Output:</strong>
                    <div className="nm-markdown" style={{ maxHeight: '400px', overflowY: 'auto', padding: '1.5rem', backgroundColor: 'var(--surface)', borderRadius: 'var(--radius-sm)', boxShadow: 'var(--nm-pressed)', border: '1px solid var(--shadow-dark)', marginTop: '0.5rem' }}>
                      <ReactMarkdown>{submitStatus.auditLog}</ReactMarkdown>
                    </div>
                  </div>
                )}
                
                {submitStatus.contract && (
                  <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--shadow-dark)' }}>
                    <button 
                      type="button" 
                      onClick={() => setShowJson(!showJson)} 
                      className="nm-button" 
                      style={{ fontSize: '0.75rem', padding: '0.5rem 1rem' }}
                    >
                      {showJson ? 'Hide Raw JSON' : 'Show Raw JSON Contract'}
                    </button>
                    {showJson && (
                      <pre style={{ marginTop: '1rem', whiteSpace: 'pre-wrap', maxHeight: '300px', overflowY: 'auto', padding: '1rem', backgroundColor: 'var(--surface)', borderRadius: 'var(--radius-sm)', boxShadow: 'var(--nm-pressed)', border: '1px solid var(--shadow-dark)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {JSON.stringify(submitStatus.contract, null, 2)}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            )}
          </form>
        </div>

        {/* Zone 3: Action Dispatch (Recent Actions) */}
        <div className="nm-card" style={{ gridColumn: 'span 2' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.25rem' }}>
            <Database className="text-primary" /> Action Dispatch
          </h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--shadow-dark)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.75rem' }}>Game</th>
                  <th style={{ padding: '0.75rem' }}>Path</th>
                  <th style={{ padding: '0.75rem' }}>Score</th>
                  <th style={{ padding: '0.75rem' }}>Cosine</th>
                  <th style={{ padding: '0.75rem' }}>Action Plan</th>
                </tr>
              </thead>
              <tbody>
                {recentActions.slice().reverse().slice(0, 5).map((action, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--shadow-dark)' }}>
                    <td style={{ padding: '0.75rem' }}>{action.game_id}</td>
                    <td style={{ padding: '0.75rem' }}>
                      <span className="nm-badge">{action.decision_path}</span>
                    </td>
                    <td style={{ padding: '0.75rem' }}>{action.intelligent_score.toFixed(3)}</td>
                    <td style={{ padding: '0.75rem' }}>{action.shap_cosine.toFixed(3)}</td>
                    <td style={{ padding: '0.75rem', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {action.action_plan}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}
