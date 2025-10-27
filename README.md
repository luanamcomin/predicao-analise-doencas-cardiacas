# Preditor de Doenças Cardíacas (Streamlit)

Aplicativo interativo em Streamlit para estimar o risco de doença cardíaca com base em atributos clínicos. O projeto cobre o pipeline end-to-end: aquisição e tratamento de dados, EDA, aprendizagem supervisionada (classificação) e não supervisionada (clusterização), com interface amigável para uso por profissionais de saúde.

- Acesse o app: https://ml-saude-luana-comin.streamlit.app
- Dataset: UCI Heart Disease (id=45)

## Sumário
- [Visão Geral](#visão-geral)
- [Como o projeto funciona (passo a passo)](#como-o-projeto-funciona-passo-a-passo)
- [Arquitetura e Organização](#arquitetura-e-organização)
- [Como Executar Localmente](#como-executar-localmente)
- [Fluxo de Dados e Tratamento](#fluxo-de-dados-e-tratamento)
- [Exploração de Dados (EDA)](#exploração-de-dados-eda)
- [Aprendizagem Supervisionada (Classificação)](#aprendizagem-supervisionada-classificação)
- [Aprendizagem Não Supervisionada (Clusterização)](#aprendizagem-não-supervisionada-clusterização)
- [Interface e Usabilidade](#interface-e-usabilidade)
- [Métricas e Validação](#métricas-e-validação)
- [Limitações e Próximos Passos](#limitações-e-próximos-passos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Licença](#licença)

## Visão Geral

- Objetivo: auxiliar na avaliação do risco de doença cardíaca a partir de dados clínicos.
- Público: profissionais de saúde e estudantes que desejam um apoio à decisão com interpretação visual.
- Escopo: estudo educacional/demonstrativo. Não substitui avaliação médica.

## Como o projeto funciona (passo a passo)

- Dados são baixados do repositório UCI via `ucimlrepo` quando necessário.
- O dataset é limpo e padronizado em `models.py`:
  - Conversão de `?` para `NaN` e coerção de tipos (ex.: `ca`, `thal`).
  - Imputação de faltantes por mediana.
  - Renomeação das colunas para português e seleção de um conjunto consistente de atributos.
- Para avaliação do classificador:
  - Os dados são divididos em treino/teste (75/25, estratificado).
  - As features são padronizadas com `StandardScaler`.
  - Um `RandomForestClassifier` é treinado com hiperparâmetros definidos.
  - Métricas e gráficos (ROC, matriz de confusão, PR, importâncias) são gerados.
- No app (Streamlit):
  - Aba EDA: visualizações exploratórias do dataset.
  - Aba Classificação: formulário "Dados do Paciente" faz predição individual usando o scaler/modelo.
  - Aba Agrupamento: KMeans com k ajustável, PCA 2D, heatmap de perfis e comparações por atributo.
- Artefatos (modelo e scaler) podem ser salvos/carregados para reuso.

Fluxo simplificado:

`Dados UCI` → `Limpeza e padronização` → `Split treino/teste` → `Scaler + RandomForest` → `Métricas e figuras` → `Interface Streamlit (EDA/Classificação/Clusterização)`

## Arquitetura e Organização

- [app.py](cci:7://file:///c:/Users/Luana/OneDrive/%C3%81rea%20de%20Trabalho/Heart-disease-predictor-main/app.py:0:0-0:0): interface Streamlit, EDA interativa, formulário do paciente, avaliação visual do modelo e clusterização.
- [models.py](cci:7://file:///c:/Users/Luana/OneDrive/%C3%81rea%20de%20Trabalho/Heart-disease-predictor-main/models.py:0:0-0:0): lógica de dados e modelagem (carregamento, pré-processamento, treino, avaliação, geração de figuras).
- Cache com `@st.cache_data` para acelerar reexecuções.
- Nomes de colunas padronizados para português, coerentes entre dataset e formulário.

## Como Executar Localmente

Pré-requisitos:
- Python 3.10+ (recomendado 3.11/3.12/3.13)
- Windows PowerShell: ative o venv com `venv\Scripts\activate` (pode ser necessário `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`)

Passos:
1. Crie e ative um ambiente virtual:
   - Windows: `python -m venv venv && venv\Scripts\activate`
2. Instale dependências:
   - `python -m pip install --upgrade pip`
   - `python -m pip install -r requirements.txt`
3. Rode o app:
   - `python -m streamlit run app.py`
4. Se precisar trocar porta:
   - `python -m streamlit run app.py --server.port 8502`

## Fluxo de Dados e Tratamento

- Fonte: UCI Heart Disease (id=45) via `ucimlrepo`.
- Limpeza:
  - Conversão de `?` para NaN, coerção de colunas não numéricas (ex.: `ca`, `thal`) para numérico.
  - Imputação de faltantes por mediana.
- Padronização de nomes (português):
  - Ex.: `age` → `Idade`, `thalach` → `Frequencia cardiaca maxima`, `trestbps` → `Pressão em reposo`.
- Target:
  - `alvo = 1` (num > 0), `alvo = 0` (ausência).

## Exploração de Dados (EDA)

- Resumo de amostras, número de atributos e taxa de positivos.
- Distribuição do alvo e estatísticas descritivas (numéricas).
- Correlação entre atributos (heatmap).
- Explorações interativas:
  - Histogramas de atributos selecionáveis (com `kde`).
  - Dispersão 2D configurável com coloração pelo alvo.

Insights esperados:
- Idade e frequência cardíaca máxima impactam o risco.
- Padrões de correlação entre pressão/colesterol e o alvo variam conforme subgrupos.

## Aprendizagem Supervisionada (Classificação)

- Modelo: RandomForestClassifier (parâmetros: `n_estimators=200`, `max_depth=10`, `min_samples_split=10`, `random_state=42`).
- Escalonamento: StandardScaler.
- Avaliação (hold-out 25% estratificado):
  - Acurácia (teste)
  - ROC-AUC (teste)
  - Matriz de confusão
  - Curva ROC
  - Importâncias de atributos (Top 10)

Uso em predição individual:
- Formulário “Dados do Paciente” na aba Classificação.
- Ao clicar “Prever Risco de Doença Cardíaca”, os dados são escalonados e enviados ao modelo; a interface exibe o resultado e o escore de risco.

## Aprendizagem Não Supervisionada (Clusterização)

- KMeans com `k` ajustável (2 a 8).
- Critérios de apoio à escolha do k:
  - Elbow (WCSS/inertia) em expander dedicado.
  - Silhouette: score médio e gráfico por cluster.
- Visualizações e interpretações:
  - PCA 2D colorido por cluster.
  - Tabela de tamanhos por cluster.
  - Centros dos clusters revertidos para escala original.
  - Heatmap de perfis em z-score por cluster.
  - Distribuição por cluster via gráfico de violino/box do atributo selecionado.
- Objetivo: observar segmentações naturais e propor hipóteses clínicas.

## Interface e Usabilidade

- Layout responsivo, com largura do container controlada para melhor leitura.
- Gráficos dimensionados para não esticar em largura total.
- Navegação por abas (EDA, Classificação, Agrupamento).
- Formulário no topo da aba de Classificação, com botão de predição e resultado logo abaixo.

## Métricas e Validação

- Hold-out estratificado (75/25) para avaliação rápida.
- Métricas principais:
  - Acurácia e ROC-AUC (robustas a desbalanceamento moderado).
  - Matriz de confusão para observar erros tipo I/II.
- Importâncias do modelo (Gini) para interpretação de variáveis preditivas.

## Limitações e Próximos Passos

Limitações:
- Avaliação em único split (hold-out).
- Sem calibração de probabilidade (pode afetar interpretação do escore).
- Sem explicabilidade local (ex.: SHAP/LIME).

Próximos passos recomendados:
- Validação mais robusta (K-fold estratificado).
- Curva Precision-Recall e F1 (se foco for classe positiva).
- Calibração (Platt/Isotônica) e ajuste de threshold.
- Explicabilidade com SHAP (global/local).
- Persistência de artefatos (modelo/escalonador) para deploy estável.
- Melhorias de EDA: análise de faltantes, outliers, segmentações por subgrupos (sexo/idade).
- Clusterização: escolha de k via Silhouette/Elbow e perfis de cluster.

## Critérios atendidos

- Aplicação prática e interativa em Streamlit demonstrando técnicas de ML supervisionada e não supervisionada em dataset real de saúde (UCI Heart Disease).
- Coleta e tratamento de dados: carregamento via `ucimlrepo`, limpeza de faltantes, coerção de tipos (`ca`, `thal`), padronização de nomes, escalonamento (`StandardScaler`).
- Modelagem supervisionada: Random Forest com avaliação (hold-out estratificado), métricas (Acurácia, ROC-AUC), PR Curve (AP), matrizes de confusão (bruta e normalizada) e importâncias de atributos.
- Modelagem não supervisionada: KMeans com seleção de `k` apoiada por Elbow e Silhouette, PCA 2D, centros na escala original, perfis em z-score e análise de distribuições por cluster.
- Interpretação e comunicação: EDA com sumário, distribuição do alvo, estatísticas, correlações e gráficos interativos; README claro com instruções e limitações; interface organizada e responsiva.