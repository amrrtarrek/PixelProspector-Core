import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
import time

try:
    with open('d:/PixelProspector-Core/02_unsupervised_ml/training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_g, y_g = data['game_features'], data['game_labels']
    X_u, y_u = data['user_features'], data['user_labels']

    sil_g = silhouette_score(X_g, y_g)
    sil_u = silhouette_score(X_u, y_u)
    
    svm_g = SVC(kernel='rbf', probability=True, random_state=42).fit(X_g, y_g)
    svm_u = SVC(kernel='rbf', probability=True, random_state=42).fit(X_u, y_u)

    print(f'Game SVM Accuracy: {accuracy_score(y_g, svm_g.predict(X_g)):.4f}')
    print(f'Game SVM F1 (macro): {f1_score(y_g, svm_g.predict(X_g), average="macro"):.4f}')
    print(f'User SVM Accuracy: {accuracy_score(y_u, svm_u.predict(X_u)):.4f}')
    print(f'User SVM F1 (macro): {f1_score(y_u, svm_u.predict(X_u), average="macro"):.4f}')
    print(f'Game Silhouette: {sil_g:.4f}')
    print(f'User Silhouette: {sil_u:.4f}')

except Exception as e:
    print('Error:', e)
