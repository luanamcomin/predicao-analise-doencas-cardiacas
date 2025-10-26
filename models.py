import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    precision_recall_curve,
    average_precision_score,
)

"""Módulo de dados/modelos: carga, pré-processamento, treino e avaliação."""

# Mapeamento de colunas inglês -> português (mantém nomes usados no app)
COL_RENAME_PT = {
    'age': 'Idade',
    'sex': 'Sexo',
    'cp': 'Tipo de dor no peito',
    'trestbps': 'Pressão em reposo',
    'chol': 'Colesterol',
    'fbs': 'Glicemia em jejum',
    'restecg': 'ECG em reposo',
    'thalach': 'Frequencia cardiaca maxima',
    'exang': 'Angina exercicio',
    'oldpeak': 'Depressao ST',
    'slope': 'Inclinacao ST',
    'ca': 'Vasos coloridos',
    'thal': 'Talassemia',
}
TARGET_RENAME_PT = {'num': 'alvo'}
# ATENÇÃO: garantir consistência com os nomes acima (acentos). "Pressão em reposo" tem acento em 'Pressão'.
FEATURE_COLS_PT = [
    'Idade', 'Sexo', 'Tipo de dor no peito', 'Pressão em reposo', 'Colesterol', 'Glicemia em jejum',
    'ECG em reposo', 'Frequencia cardiaca maxima', 'Angina exercicio', 'Depressao ST', 'Inclinacao ST',
    'Vasos coloridos', 'Talassemia'
]


def load_processed_dataset():
    """Carrega o dataset UCI, trata dados e retorna X (features) e y (alvo binário)."""
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    X.replace('?', np.nan, inplace=True)
    X['ca'] = pd.to_numeric(X['ca'], errors='coerce')
    X['thal'] = pd.to_numeric(X['thal'], errors='coerce')
    X.fillna(X.median(), inplace=True)

    # Renomear
    X = X.rename(columns=COL_RENAME_PT)
    y = y.rename(columns=TARGET_RENAME_PT)
    y_binary = (y['alvo'] > 0).astype(int)

    # Ordenar/filtrar colunas conforme usado no treino
    X = X[FEATURE_COLS_PT]
    return X, y_binary


def train_model():
    """Treina RandomForest com StandardScaler e retorna (model, scaler)."""
    X, y = load_processed_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_params = {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 200}
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_scaled, y)
    return model, scaler


def save_artifacts(model, scaler, path='artifacts'):
    """Salva modelo e scaler em arquivos pickle."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)


def load_artifacts(path='artifacts'):
    """Carrega artefatos; em falha remove arquivos e retorna (None, None)."""
    model_path = os.path.join(path, 'model.pkl')
    scaler_path = os.path.join(path, 'scaler.pkl')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception:
            try:
                os.remove(model_path)
            except Exception:
                pass
            try:
                os.remove(scaler_path)
            except Exception:
                pass
            return None, None
    return None, None


def evaluate_model():
    """Executa holdout, treina e retorna métricas e figuras para o app (sem Streamlit)."""
    X, y = load_processed_dataset()

    scaler_eval = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_train_s = scaler_eval.fit_transform(X_train)
    X_test_s = scaler_eval.transform(X_test)

    best_params = {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 200}
    clf = RandomForestClassifier(random_state=42, **best_params)
    clf.fit(X_train_s, y_train)

    y_prob = clf.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float('nan')

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)

    # Figuras
    # Importâncias
    fig_imp, ax_imp = plt.subplots(figsize=(8, 7))
    importances.tail(10).plot(kind='barh', ax=ax_imp, color='#1f77b4')
    ax_imp.set_title('Top 10 Importâncias de Atributos', fontsize=20)
    ax_imp.set_xlabel('Importância (Gini)', fontsize=18)
    ax_imp.tick_params(axis='both', labelsize=14)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(3, 2.4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
    ax_cm.set_title('Matriz de Confusão (teste)')
    ax_cm.set_xlabel('Predito')
    ax_cm.set_ylabel('Real')

    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(3, 2.4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
    ax_roc.set_title('Curva ROC (teste)')

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig_pr, ax_pr = plt.subplots(figsize=(3, 2.4))
    ax_pr.plot(recall, precision, color='#1f77b4')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'Curva Precision-Recall (AP={ap:.3f})')

    # Matriz de confusão normalizada
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
    fig_cm_norm, ax_cmn = plt.subplots(figsize=(3, 2.4))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax_cmn)
    ax_cmn.set_title('Matriz de Confusão (normalizada)')
    ax_cmn.set_xlabel('Predito')
    ax_cmn.set_ylabel('Real')

    return {
        'acc': acc,
        'auc': auc,
        'importances': importances,
        'fig_imp': fig_imp,
        'fig_cm': fig_cm,
        'fig_cm_norm': fig_cm_norm,
        'fig_roc': fig_roc,
        'fig_pr': fig_pr,
    }
