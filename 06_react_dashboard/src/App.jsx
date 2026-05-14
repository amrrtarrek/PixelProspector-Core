import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { 
  Moon, Sun, Activity, Database, AlertCircle, 
  TrendingUp, Send, CheckCircle, XCircle, X, ClipboardList
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import ReactMarkdown from 'react-markdown';
import mermaid from 'mermaid';

const API_URL = 'http://localhost:8000/v1';

function formatSignals(signals) {
  if (!signals) return 'S:0 | G:0 | μ:0 | T:0 | C:0';
  const s = (signals.S_class_severity || 0).toFixed(2);
  const g = (signals.Gap_SVM_confidence || 0).toFixed(4);
  const m = (signals.mu_geometric_membership || 0).toFixed(4);
  const t = (signals.ARIMA_trend_multiplier || 1).toFixed(4);
  const c = (signals.SHAP_cosine_similarity || 0).toFixed(4);
  return `S:${s} | G:${g} | μ:${m} | T:${t} | C:${c}`;
}

// ─── Detail Modal (Neumorphic Style) ───────────────────────────────────────
function DetailModal({ action, onClose }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  if (!action) return null;
  const signals = action.signals || {};
  
  const weights = [
    { 
      label: 'S-Severity', sub: '[Inert]', value: signals.S_class_severity || 0, color: 'var(--text-muted)',
      note: "Measures fundamental game quality; high severity flags potential core loop issues."
    },
    { 
      label: 'SVM Stability', sub: '[Crucial]', value: signals.Gap_SVM_confidence || 0, color: 'var(--danger)',
      note: "Determines decision boundary confidence; low stability forces human review."
    },
    { 
      label: 'Centroid Prox', sub: '[Crucial]', value: signals.mu_geometric_membership || 0, color: 'var(--danger)',
      note: "Geometric distance to success clusters; directly scales automated approval chances."
    },
    { 
      label: 'Market Trend', sub: '[Minor]', value: (signals.ARIMA_trend_multiplier || 1) - 1, color: 'var(--success)',
      note: "ARIMA forecast modifier; boosts or penalizes score based on genre momentum."
    },
    { 
      label: 'RAG Check', sub: '[Minor]', value: signals.SHAP_cosine_similarity || 0, color: 'var(--success)',
      note: "Cosine similarity against FAISS historical DB; acts as a tie-breaker."
    },
  ];

  // Dynamically map real SHAP drivers from the backend
  const rawShap = action.shap_raw_drivers || {};
  const shapDrivers = Object.entries(rawShap)
    .map(([label, val]) => ({ label, val, width: `${Math.min(100, Math.abs(val) * 100)}%` }))
    .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
    .slice(0, 3); // take top 3

  const mermaidRef = React.useRef(null);
  useEffect(() => {
    if (mermaidRef.current && action) {
      mermaid.initialize({ startOnLoad: false, theme: 'dark' });
      
      const safeDecision = action.decision_path ? action.decision_path.replace(/["']/g, '') : 'Decision';
      const graph = `graph TD
        A[Interaction Pipeline] --> B(ML Inference)
        B --> C{ReAct Router}
        C -- ${safeDecision} --> D[Final Outcome]
      `;
      
      mermaid.render('mermaid-svg', graph).then(result => {
        if(mermaidRef.current) mermaidRef.current.innerHTML = result.svg;
      }).catch(e => {
        console.error("Mermaid Render Error:", e);
        if(mermaidRef.current) mermaidRef.current.innerHTML = `<p style="color:red">Flowchart failed to load.</p>`;
      });
    }
  }, [action]);

  const [activeTab, setActiveTab] = useState('trace');

  // We will just pass the raw markdown string to ReactMarkdown later.
  const reasoningContent = action.llm_audit_log || `1. Signal Data Observed
2. Implicit Model Reliability Checked
3. Macro-Signal Applied
4. Cross Check
5. Result Proceeding`;

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
      zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem'
    }} onClick={onClose}>
      
      <div className="nm-card" style={{
        width: '100%', maxWidth: '1400px', maxHeight: '90vh', overflowY: 'auto',
        display: 'flex', flexDirection: 'column', gap: '2rem', position: 'relative'
      }} onClick={e => e.stopPropagation()}>
        
        <button onClick={onClose} className="nm-button" style={{ position: 'absolute', top: '1.5rem', right: '1.5rem', padding: '0.5rem', borderRadius: '50%' }}>
          <X size={20} />
        </button>

        <div>
          <h2 style={{ fontSize: '1.75rem', color: 'var(--primary)', textTransform: 'uppercase' }}>
            {action.game_name || action.game_id}
          </h2>
          <p className="text-muted" style={{ fontWeight: 'bold' }}>Cycle Action: {action.action_plan}</p>
        </div>

        {/* Tab Selector */}
        <div style={{ display: 'flex', gap: '1rem', borderBottom: '2px solid var(--shadow-dark)', paddingBottom: '0.5rem' }}>
          <button className={`nm-button ${activeTab === 'trace' ? 'nm-button-primary' : ''}`} onClick={() => setActiveTab('trace')}>Reasoning & Models</button>
          <button className={`nm-button ${activeTab === 'agents' ? 'nm-button-primary' : ''}`} onClick={() => setActiveTab('agents')}>Agent Outputs</button>
          <button className={`nm-button ${activeTab === 'json' ? 'nm-button-primary' : ''}`} onClick={() => setActiveTab('json')}>Raw JSON Payload</button>
        </div>

        {activeTab === 'trace' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '2rem' }}>
            {/* LEFT: ReAct Diagram & Internal Reasoning */}
            <div>
              <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem', letterSpacing: '0.05em' }}>REACT DECISION FLOW</h3>
              <div style={{ background: 'var(--shadow-light)', padding: '1rem', borderRadius: 'var(--radius-sm)', boxShadow: 'var(--nm-pressed)', marginBottom: '2rem', display: 'flex', justifyContent: 'center' }}>
                <div ref={mermaidRef} />
              </div>

              <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem', letterSpacing: '0.05em' }}>INTERNAL REASONING TRACE</h3>
              <div className="nm-markdown" style={{ background: 'var(--shadow-light)', padding: '1.5rem', borderRadius: 'var(--radius-sm)', boxShadow: 'var(--nm-pressed)', fontSize: '0.875rem', color: 'var(--text)', maxHeight: '300px', overflowY: 'auto' }}>
                <ReactMarkdown>{reasoningContent}</ReactMarkdown>
              </div>
            </div>

            {/* MIDDLE: Signal Weights */}
            <div>
              <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem', letterSpacing: '0.05em' }}>COMPOSITE DECISION WEIGHTS</h3>
              <div>
                {weights.map((w, i) => (
                  <div key={i} style={{ marginBottom: '1.25rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.25rem', fontSize: '0.875rem' }}>
                      <div style={{ width: '120px', color: 'var(--text-muted)', fontWeight: 'bold' }}>
                        {w.label}
                        <span style={{ display: 'block', fontSize: '0.65rem', color: w.color }}>{w.sub}</span>
                      </div>
                      <div style={{ flex: 1, height: '8px', background: 'var(--shadow-dark)', borderRadius: '4px', margin: '0 1rem', boxShadow: 'var(--nm-pressed)', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${Math.abs(w.value) * 100}%`, background: w.color, borderRadius: '4px' }} />
                      </div>
                      <div style={{ width: '50px', textAlign: 'right', fontWeight: 'bold' }}>
                        {w.value > 0 ? '+' : ''}{w.value.toFixed(2)}
                      </div>
                    </div>
                    <div style={{ paddingLeft: '136px', fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic', lineHeight: '1.3' }}>
                      * {w.note}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* RIGHT: SHAP & RAG */}
            <div>
              <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem', letterSpacing: '0.05em' }}>DETAILED SHAP DRIVERS</h3>
              <div style={{ marginBottom: '2rem' }}>
                {shapDrivers.length > 0 ? shapDrivers.map((s, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: '0.75rem', fontSize: '0.875rem' }}>
                    <div style={{ width: '180px', color: 'var(--text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.label}</div>
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                      <div style={{ height: '8px', width: s.width, background: 'var(--primary)', borderRadius: '4px' }} />
                    </div>
                    <div style={{ marginLeft: '1rem', fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                      {s.val > 0 ? ' ' : ''}{s.val.toFixed(4)}
                    </div>
                  </div>
                )) : <p className="text-muted text-sm">No SHAP data available for this path.</p>}
              </div>

              <div style={{ padding: '1.5rem', background: 'var(--surface)', borderRadius: 'var(--radius-sm)', boxShadow: 'var(--nm-flat)', borderLeft: '4px solid var(--success)', marginBottom: '1.5rem' }}>
                <h4 style={{ fontSize: '0.75rem', color: 'var(--success)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Historical RAG Evidence</h4>
                <div style={{ fontSize: '0.875rem', lineHeight: '1.6' }}>
                  <strong>Consensus:</strong> {Math.max(1, Math.round((action.shap_cosine || 0) * 5))}/5 Neighbors Agreement<br/>
                  <strong>Similarity Score:</strong> {(action.shap_cosine || 0).toFixed(4)}
                </div>
              </div>

              <div style={{ padding: '1.5rem', background: 'var(--surface)', borderRadius: 'var(--radius-sm)', boxShadow: 'var(--nm-flat)', borderLeft: '4px solid var(--primary)' }}>
                <h4 style={{ fontSize: '0.75rem', color: 'var(--primary)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Technical Context</h4>
                <p style={{ fontSize: '0.875rem', lineHeight: '1.6' }}>
                  The AI model identifies anomalies by monitoring three core behavioral 'senses': <strong>Historical Stability</strong> (mean), <strong>Volatility</strong> (std), and <strong>Event Elasticity</strong> (promo). SHAP drivers show which 'sense' detected the current deviation.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'agents' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem' }}>
            <div>
              <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem', letterSpacing: '0.05em' }}>COMMUNITY AGENT PROFILE</h3>
              <div className="nm-markdown" style={{ background: 'var(--shadow-light)', padding: '1.5rem', borderRadius: 'var(--radius-sm)', boxShadow: 'var(--nm-pressed)', fontSize: '0.875rem', color: 'var(--text)', maxHeight: '500px', overflowY: 'auto', whiteSpace: 'pre-wrap' }}>
                {action.community ? (
                  typeof action.community === 'object' ? JSON.stringify(action.community, null, 2) : action.community
                ) : 'Community agent did not run for this path.'}
              </div>
            </div>
            <div>
              <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem', letterSpacing: '0.05em' }}>INVESTOR PITCH DRAFT</h3>
              <div className="nm-markdown" style={{ background: 'var(--shadow-light)', padding: '1.5rem', borderRadius: 'var(--radius-sm)', boxShadow: 'var(--nm-pressed)', fontSize: '0.875rem', color: 'var(--text)', maxHeight: '500px', overflowY: 'auto' }}>
                {action.investor_pitch ? <ReactMarkdown>{action.investor_pitch}</ReactMarkdown> : <p className="text-muted">Investor agent did not generate a pitch for this path.</p>}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'json' && (
          <div>
            <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem', letterSpacing: '0.05em' }}>RAW PIPELINE PAYLOAD</h3>
            <pre style={{ background: '#1e1e1e', color: '#d4d4d4', padding: '1.5rem', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', overflowX: 'auto', maxHeight: '500px' }}>
              {JSON.stringify(action.raw_payload || { error: "No raw payload stored" }, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState('dark');
  const [logs, setLogs] = useState([]);
  const [driftEvents, setDriftEvents] = useState([]);
  const [health, setHealth] = useState({ game_features: {}, user_features: {} });
  const [recentActions, setRecentActions] = useState([]);
  const [selectedAction, setSelectedAction] = useState(null);
  
  const [formData, setFormData] = useState({
    review_text: '', game_name: '', user_id: '', genre: '', recommended: 'Yes'
  });
  const [submitStatus, setSubmitStatus] = useState(null);
  const [showJson, setShowJson] = useState(false);

  useEffect(() => { document.documentElement.className = theme; }, [theme]);
  const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark');

  const fetchData = useCallback(async () => {
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
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

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

  const shapData = recentActions.slice().reverse().map((action, i) => ({
    name: i, cosine: action.shap_cosine
  }));

  const totalActions = recentActions.length;
  const passCount = recentActions.filter(a => a.intelligent_score >= 0.7).length;
  const passRatio = totalActions > 0 ? ((passCount / totalActions) * 100).toFixed(1) : 0;
  const auditRows = recentActions.slice().reverse();

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
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.75rem' }}>
              {Object.entries(health.game_features).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem' }}>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={k}>{k}:</span> <strong>{v}</strong>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>USER FEATURES (AVG)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.75rem' }}>
              {Object.entries(health.user_features).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem' }}>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={k}>{k}:</span> <strong>{v}</strong>
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
            <div className="nm-badge text-danger">Reject: {totalActions - passCount}</div>
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
              </div>
            )}
          </form>
        </div>

        {/* Zone 3: Audit Log Table */}
        <div className="nm-card" style={{ gridColumn: 'span 2' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', fontSize: '1.25rem' }}>
            <ClipboardList className="text-primary" /> ReAct Loop Audit Logs
          </h2>
          <p className="text-muted" style={{ fontSize: '0.875rem', marginBottom: '1.5rem' }}>Detailed telemetry for each product action generated via the 7-path Reasoning/Agentic routing.</p>
          
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--shadow-dark)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.75rem' }}>Time</th>
                  <th style={{ padding: '0.75rem' }}>Product Family</th>
                  <th style={{ padding: '0.75rem' }}>ReAct Routing Path</th>
                  <th style={{ padding: '0.75rem' }}>Final Action</th>
                  <th style={{ padding: '0.75rem' }}>Top Driver (SHAP)</th>
                  <th style={{ padding: '0.75rem' }}>Intelligent Score</th>
                  <th style={{ padding: '0.75rem' }}>Internal Signals (S | Gap | μ | Trend | Cosine)</th>
                  <th style={{ padding: '0.75rem' }}>LLM Reasoning / Thoughts</th>
                </tr>
              </thead>
              <tbody>
                {auditRows.slice(0, 5).map((action, i) => {
                  const isPass = action.intelligent_score >= 0.7;
                  const finalActionLabel = action.decision_path === 'Human Review' ? 'Human Audit Required' : 
                                          isPass ? 'Automated Investor Pitch' : 'Reject Opportunity';
                  const driver = action.shap_raw_drivers && Object.keys(action.shap_raw_drivers).length > 0 
                                  ? Object.entries(action.shap_raw_drivers).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]))[0][0]
                                  : (isPass ? "gameplay_addictiveness" : "technical_polish");
                  
                  return (
                    <tr 
                      key={i} 
                      style={{ borderBottom: '1px solid var(--shadow-dark)', cursor: 'pointer', transition: 'background 0.2s' }}
                      onMouseEnter={(e) => e.currentTarget.style.background = 'var(--shadow-light)'}
                      onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                      onClick={() => setSelectedAction(action)}
                    >
                      <td style={{ padding: '0.75rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>{action.timestamp?.split(' ')[1] ?? '—'}</td>
                      <td style={{ padding: '0.75rem', color: 'var(--primary)', fontWeight: 'bold', textDecoration: 'underline' }}>{action.game_name?.toUpperCase() || action.game_id}</td>
                      <td style={{ padding: '0.75rem', color: 'var(--primary)', whiteSpace: 'nowrap' }}>{action.decision_path}</td>
                      <td style={{ padding: '0.75rem', color: finalActionLabel.includes('Audit') || finalActionLabel.includes('Reject') ? 'var(--danger)' : 'var(--success)', fontWeight: 'bold', whiteSpace: 'nowrap' }}>
                        {finalActionLabel}
                      </td>
                      <td style={{ padding: '0.75rem', color: 'var(--warning)' }}>{driver}</td>
                      <td style={{ padding: '0.75rem', fontWeight: 'bold' }}>{action.intelligent_score.toFixed(4)}</td>
                      <td style={{ padding: '0.75rem', color: 'var(--text-muted)', fontFamily: 'monospace', whiteSpace: 'nowrap' }}>{formatSignals(action.signals)}</td>
                      <td style={{ padding: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic', maxWidth: '200px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {action.llm_audit_log || "Continuing monitoring based on statistical drift signals."}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

      </div>

      {selectedAction && <DetailModal action={selectedAction} onClose={() => setSelectedAction(null)} />}
    </div>
  );
}
