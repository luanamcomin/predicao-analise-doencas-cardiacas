import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
from models import (
    load_processed_dataset as load_ds,
    train_model as model_train,
    evaluate_model as model_evaluate,
    FEATURE_COLS_PT,
    load_artifacts,
    save_artifacts,
)

# Configuração da página (deve ser o primeiro comando do Streamlit)
st.set_page_config(page_title="Preditor de Doenças Cardíacas", layout="wide")

# Estilos globais (UX)
st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; }
    img { max-width: 600px !important; height: auto !important; }
    div[data-testid=\"stHorizontalBlock\"] { gap: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cache/Treino do modelo e carregamento de artefatos
@st.cache_data
def train_model():
    """Encapsula o treino do modelo para uso com cache do Streamlit."""
    return model_train()

# Carrega artefatos; se ausentes/corrompidos, re-treina e salva
model, scaler = load_artifacts()
if model is None or scaler is None:
    model, scaler = train_model()
    save_artifacts(model, scaler)

# Dados processados (cache)
@st.cache_data
def _load_processed_dataset():
    """Carrega X e y já tratados para EDA/clusterização."""
    return load_ds()

# Seção de avaliação supervisionada
def evaluate_model_section():
    """Exibe métricas e gráficos do modelo supervisionado no conjunto de teste."""
    st.subheader('Aprendizado Supervisionado — Classificação')
    results = model_evaluate()
    acc = results['acc']
    auc = results['auc']
    fig_imp = results['fig_imp']
    fig_cm = results['fig_cm']
    fig_roc = results['fig_roc']
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('Acurácia (teste)', f"{acc:.3f}")
        st.metric('ROC-AUC (teste)', f"{auc:.3f}")
    with c2:
        st.pyplot(fig_imp, width='content', clear_figure=True)
    with c3:
        st.pyplot(results['fig_pr'], width='content', clear_figure=True)
    st.pyplot(fig_cm, width='content', clear_figure=True)
    st.pyplot(results['fig_cm_norm'], width='content', clear_figure=True)
    st.pyplot(fig_roc, width='content', clear_figure=True)

# Seção de clusterização (não supervisionado)
def clustering_section():
    """Aplica KMeans, auxilia na escolha de k e interpreta clusters."""
    st.subheader('Aprendizado Não Supervisionado — Clusterização (KMeans)')
    X, _ = _load_processed_dataset()
    scaler_c = StandardScaler()
    Xs = scaler_c.fit_transform(X)

    # Seleção de k
    k = st.slider('Número de clusters (k)', 2, 8, 3)
    with st.expander('Escolha do k (Elbow)'):
        ks = list(range(2, 9))
        wcss = []
        for kk in ks:
            km_kk = KMeans(n_clusters=kk, random_state=42, n_init=10)
            km_kk.fit(Xs)
            wcss.append(km_kk.inertia_)
        fig_elbow, ax_elbow = plt.subplots(figsize=(4, 3))
        ax_elbow.plot(ks, wcss, marker='o')
        ax_elbow.set_xlabel('k')
        ax_elbow.set_ylabel('WCSS (inertia)')
        ax_elbow.set_title('Elbow method')
        st.pyplot(fig_elbow, width='content', clear_figure=True)

    # KMeans e métricas
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    sil_avg = silhouette_score(Xs, labels)
    with st.expander(f'Silhouette (k={k}) — score médio: {sil_avg:.3f}'):
        sample_sil = silhouette_samples(Xs, labels)
        fig_sil, ax_sil = plt.subplots(figsize=(4, 3))
        y_lower = 10
        for i in sorted(set(labels)):
            ith_sil = sample_sil[labels == i]
            ith_sil.sort()
            size_i = ith_sil.shape[0]
            y_upper = y_lower + size_i
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil, alpha=0.7)
            ax_sil.text(-0.05, y_lower + 0.5 * size_i, str(i))
            y_lower = y_upper + 10
        ax_sil.axvline(x=sil_avg, color='red', linestyle='--')
        ax_sil.set_xlabel('Coeficiente de Silhouette')
        ax_sil.set_ylabel('Clusters')
        ax_sil.set_title('Gráfico de Silhouette')
        st.pyplot(fig_sil, width='content', clear_figure=True)

    # Visualização PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)
    fig_scatter, ax_scatter = plt.subplots(figsize=(4, 3))
    palette = sns.color_palette('tab10', n_colors=k)
    for lab in range(k):
        ax_scatter.scatter(X2[labels == lab, 0], X2[labels == lab, 1], s=14, color=palette[lab], label=f'Cluster {lab}')
    ax_scatter.set_title('PCA 2D dos Clusters')
    ax_scatter.set_xlabel('PC1')
    ax_scatter.set_ylabel('PC2')
    ax_scatter.legend(loc='best', fontsize='x-small')
    st.pyplot(fig_scatter, width='content', clear_figure=True)

    # Tamanho e centros
    st.write('Tamanho dos clusters:')
    st.write(pd.Series(labels).value_counts().sort_index().rename('Contagem').to_frame())

    centers_orig = pd.DataFrame(
        scaler_c.inverse_transform(km.cluster_centers_),
        columns=X.columns
    )
    st.markdown('Centros dos clusters (escala original):')
    st.dataframe(centers_orig.round(2))

    # Perfis (z-score) e distribuição por atributo
    group_means = pd.DataFrame(Xs, columns=X.columns).groupby(labels).mean()
    z_profile = group_means
    fig_hm, ax_hm = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(z_profile, cmap='vlag', center=0, annot=False, ax=ax_hm)
    ax_hm.set_xlabel('Atributos')
    ax_hm.set_ylabel('Cluster')
    ax_hm.set_title('Perfis de clusters (z-score)')
    st.pyplot(fig_hm, width='content', clear_figure=True)

    feat = st.selectbox('Atributo para distribuição por cluster', X.columns.tolist())
    fig_vio, ax_vio = plt.subplots(figsize=(4, 3))
    tmp = X.copy()
    tmp['cluster'] = labels
    sns.violinplot(data=tmp, x='cluster', y=feat, palette='tab10', ax=ax_vio, inner='box')
    ax_vio.set_title(f'Distribuição por cluster: {feat}')
    st.pyplot(fig_vio, width='content', clear_figure=True)

# Seção de EDA
def eda_section():
    """Explora o dataset com métricas, correlações e gráficos interativos."""
    st.subheader('Análise Exploratória dos Dados (EDA)')

    # Dados para EDA
    X, y = _load_processed_dataset()

    # Resumo geral
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('Total de amostras', f"{X.shape[0]}")
    with c2:
        st.metric('Número de atributos', f"{X.shape[1]}")
    with c3:
        st.metric('Taxa de positivos (alvo=1)', f"{(y.mean()*100):.1f}%")

    # Distribuição do alvo
    st.markdown('**Distribuição do alvo (num>0 → 1):**')
    vc = y.value_counts().sort_index()
    fig_t, ax_t = plt.subplots(figsize=(3.5, 2.5))
    sns.barplot(x=vc.index.astype(str), y=vc.values, hue=vc.index.astype(str), ax=ax_t, palette='Blues', legend=False)
    ax_t.set_xlabel('Classe')
    ax_t.set_ylabel('Contagem')
    st.pyplot(fig_t, width='content', clear_figure=True)

    # Amostra de dados
    st.markdown('**Amostra dos dados:**')
    st.dataframe(pd.concat([X.head(10), y.head(10).rename('target')], axis=1))

    # Estatísticas descritivas
    st.markdown('**Estatísticas descritivas (numéricas):**')
    st.dataframe(X.describe().T)

    # Faltantes por coluna
    st.markdown('**Valores faltantes por coluna:**')
    miss = X.isna().sum()
    st.dataframe(miss[miss > 0].sort_values(ascending=False))

    # Heatmap de correlação reduzido
    st.markdown('**Correlação entre atributos:**')
    fig_corr, ax_corr = plt.subplots(figsize=(4, 3))
    sns.heatmap(X.corr(numeric_only=True), cmap='RdBu_r', center=0, ax=ax_corr)
    ax_corr.set_title('Correlação entre atributos')
    st.pyplot(fig_corr, width='content', clear_figure=True)

    # Explorações interativas
    st.markdown('**Explorações interativas:**')
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    with st.expander('Histogramas e Boxplots (selecione múltiplos)', expanded=False):
        defaults = [c for c in ['Idade', 'Colesterol', 'Frequencia cardiaca maxima'] if c in num_cols]
        sel_cols = st.multiselect('Atributos numéricos', num_cols, default=defaults)

        n = max(1, min(3, len(sel_cols)))
        for i in range(0, len(sel_cols), n):
            row_cols = sel_cols[i:i+n]
            cols = st.columns(len(row_cols))
            for col_name, area in zip(row_cols, cols):
                with area:
                    fig_h, ax_h = plt.subplots(figsize=(3, 2.2))
                    sns.histplot(X[col_name], bins=15, kde=True, ax=ax_h)
                    ax_h.set_title(f'Distribuição: {col_name}')
                    st.pyplot(fig_h, width='content', clear_figure=True)

                    fig_b, ax_b = plt.subplots(figsize=(3, 2.2))
                    sns.boxplot(x=X[col_name], ax=ax_b, color='#4c78a8')
                    ax_b.set_title(f'Boxplot: {col_name}')
                    st.pyplot(fig_b, width='content', clear_figure=True)
    with st.expander('Dispersão 2D (escolha eixos)', expanded=False):
        x_feat = st.selectbox('Eixo X', num_cols, index=num_cols.index('Idade') if 'Idade' in num_cols else 0)
        y_feat = st.selectbox('Eixo Y', num_cols, index=num_cols.index('Frequencia cardiaca maxima') if 'Frequencia cardiaca maxima' in num_cols else 1)

        fig_s, ax_s = plt.subplots(figsize=(4, 3))
        palette = {0:'#4c78a8', 1:'#f58518'}
        ax_s.scatter(X[x_feat], X[y_feat], c=y.map(palette), s=12, alpha=0.7)
        ax_s.set_xlabel(x_feat)
        ax_s.set_ylabel(y_feat)
        ax_s.set_title('Relação entre características (colorido pelo alvo)')
        st.pyplot(fig_s, width='content', clear_figure=True)

# Interface da aplicação
st.title(' Preditor de Doenças Cardíacas')
st.markdown("""
Este aplicativo estima a probabilidade de um paciente ter doença cardíaca com base em informações clínicas.
Isto é uma **ferramenta de apoio à decisão** e não substitui a avaliação médica profissional.
""")

# Formulário do paciente (usado na aba Classificação)
def user_input_features():
    """Coleta os dados do paciente e retorna DataFrame com as features padronizadas pelo projeto."""
    # Layout em colunas
    c1, c2, c3 = st.columns(3)
    with c1:
        idade = st.number_input('Idade', 1, 100, 50)
        colesterol = st.number_input('Colesterol Sérico (mg/dL)', 100, 600, 200)
        depressao_st = st.slider('Depressão do Segmento ST (oldpeak)', 0.0, 6.2, 1.0, 0.1)
        vasos_coloridos = st.selectbox('Nº de Vasos Principais Corados por Fluoroscopia', (0, 1, 2, 3, 4))
    with c2:
        sexo = st.selectbox('Sexo', ('Masculino', 'Feminino'))
        glicemia_jejum_txt = st.selectbox('Glicemia de Jejum > 120 mg/dL', ('Não', 'Sim'))
        inclinacao_st = st.selectbox('Inclinação do Segmento ST no Pico do Exercício', (0, 1, 2), format_func=lambda x: {0: 'Ascendente', 1: 'Plano', 2: 'Descendente'}[x])
        talassemia = st.selectbox('Talassemia', (0, 1, 2, 3), format_func=lambda x: {0: 'Desconhecido', 1: 'Normal', 2: 'Defeito Fixo', 3: 'Defeito Reversível'}[x])
    with c3:
        tipo_dor_peito = st.selectbox('Tipo de Dor no Peito', (0, 1, 2, 3), format_func=lambda x: {0: 'Angina Típica', 1: 'Angina Atípica', 2: 'Dor Não Anginosa', 3: 'Assintomático'}[x])
        pressao_reposo = st.number_input('Pressão Arterial em Repouso (mmHg)', 80, 200, 120)
        ecg_repouso = st.selectbox('Resultado do ECG de Repouso', (0, 1, 2), format_func=lambda x: {0: 'Normal', 1: 'Anormalidade de ST-T', 2: 'Hipertrofia Ventricular Esquerda'}[x])
        frequencia_cardiaca_maxima = st.number_input('Frequencia cardiaca maxima', 60, 220, 150)
        angina_exercicio_txt = st.selectbox('Angina Induzida por Exercício', ('Não', 'Sim'))

    sexo_num = 1 if sexo == 'Masculino' else 0
    glicemia_jejum = 1 if glicemia_jejum_txt == 'Sim' else 0
    angina_exercicio = 1 if angina_exercicio_txt == 'Sim' else 0

    data = {
        'Idade': idade,
        'Sexo': sexo_num,
        'Tipo de dor no peito': tipo_dor_peito,
        'Pressão em reposo': pressao_reposo,
        'Colesterol': colesterol,
        'Glicemia em jejum': glicemia_jejum,
        'ECG em reposo': ecg_repouso,
        'Frequencia cardiaca maxima': frequencia_cardiaca_maxima,
        'Angina exercicio': angina_exercicio,
        'Depressao ST': depressao_st,
        'Inclinacao ST': inclinacao_st,
        'Vasos coloridos': vasos_coloridos,
        'Talassemia': talassemia
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Navegação entre seções
nav = st.sidebar.radio('Navegação', ['Análise Exploratória', 'Classificação (Supervisionado)', 'Agrupamento (Não Supervisionado)'], index=0)

if nav == 'Análise Exploratória':
    eda_section()
elif nav == 'Classificação (Supervisionado)':
    # Formulário de Dados do Paciente no início da aba
    st.subheader('Dados do Paciente')
    input_df = user_input_features()
    st.write(input_df)

    # Predição individual (botão e resultado logo abaixo do formulário)
    if st.button('**Prever Risco de Doença Cardíaca**'):
        # Garantir ordem e nomes das colunas conforme treino
        input_ordered = input_df.reindex(columns=FEATURE_COLS_PT)
        input_scaled = scaler.transform(input_ordered)
        prediction_proba = model.predict_proba(input_scaled)
        prediction = model.predict(input_scaled)

        risk_score = prediction_proba[0][1] * 100
        st.subheader('Resultado da Predição')
        if prediction[0] == 1:
            st.error('**Alto risco de Doença Cardíaca**')
        else:
            st.success('**Baixo risco de Doença Cardíaca**')
        st.metric(label='Escore de Risco do Paciente', value=f"{risk_score:.2f}%")
        st.write("""
        O escore de risco representa a confiança do modelo na predição. Quanto maior o escore, maior a probabilidade de doença cardíaca.
        Esta informação deve complementar, e não substituir, a avaliação médica profissional.
        """)

    # Pipeline e avaliação após o resultado
    with st.expander('Ver etapas do pipeline de dados', expanded=False):
        st.markdown('- **Tratamento de faltantes:** imputação por mediana para colunas numéricas')
        st.markdown('- **Conversão de tipos:** `ca` e `thal` convertidos para numérico')
        st.markdown('- **Escalonamento:** StandardScaler para padronização antes do modelo')

    # Avaliação do modelo em conjunto de teste
    evaluate_model_section()
else:
    clustering_section()