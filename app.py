import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Carregar o modelo e outros objetos salvos
@st.cache_resource
def load_models():
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    smote = joblib.load('smote.joblib')
    return model, scaler, smote

# Carregar os dados estatísticos
@st.cache_data
def load_statistics():
    desc_stats = pd.read_csv('descriptive_statistics.csv')
    outlier_stats = pd.read_csv('outlier_statistics.csv')
    model_metrics = pd.read_csv('model_metrics.csv')
    return desc_stats, outlier_stats, model_metrics

# Carregar os dados originais
@st.cache_data
def load_original_data():
    df = pd.read_csv('Diabetes.csv')
    return df

# Função para fazer previsões
def predict_diabetes(model, scaler, input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    return prediction[0], probability[0]

# Carregar os modelos e estatísticas
model, scaler, smote = load_models()
desc_stats, outlier_stats, model_metrics = load_statistics()
df = load_original_data()

# Título da aplicação
st.title("🏥 Diabetes Prediction Dashboard")

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.radio("Selecione uma página:", 
                       ["Predição", 
                        "Análise de Dados", 
                        "Análise de Correlação",
                        "Análise de Outliers",
                        "Processo de Mineração",
                        "Métricas do Modelo"])

if page == "Predição":
    st.header("Predição de Diabetes")
    
    # Formulário de entrada
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Número de gestações", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glicose (mg/dL)", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Pressão sanguínea (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Espessura da pele (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulina (mu U/ml)", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("IMC (kg/m²)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input("Função de pedigree de diabetes", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        age = st.number_input("Idade", min_value=0, max_value=120, value=30)
    
    # Botão para fazer a previsão
    if st.button("Prever"):
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })
        
        prediction, probability = predict_diabetes(model, scaler, input_data)
        
        # Exibir resultados
        st.subheader("Resultado da Predição")
        if prediction == 1:
            st.error(f"Risco de Diabetes: {probability[1]*100:.2f}%")
            st.warning("Recomenda-se consultar um médico para avaliação.")
        else:
            st.success(f"Risco de Diabetes: {probability[1]*100:.2f}%")
            st.info("Risco baixo de diabetes. Mantenha hábitos saudáveis.")

elif page == "Análise de Dados":
    st.header("Análise de Dados")
    
    # 1. Distribuição das Classes
    st.subheader("Distribuição das Classes")
    class_dist = df['Outcome'].value_counts()
    
    # Gráfico de pizza e barras lado a lado
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=class_dist.values,
            names=['Não Diabético', 'Diabético'],
            title='Distribuição de Pacientes Diabéticos vs Não Diabéticos',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig)
    
    with col2:
        fig = px.bar(
            x=['Não Diabético', 'Diabético'],
            y=class_dist.values,
            title='Contagem de Pacientes por Classe',
            color=['Não Diabético', 'Diabético'],
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig)
    
    # Estatísticas da distribuição
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Pacientes", len(df))
    with col2:
        st.metric("Pacientes Não Diabéticos", class_dist[0])
    with col3:
        st.metric("Pacientes Diabéticos", class_dist[1])
    
    # 2. Análise de Features
    st.subheader("Análise de Features")
    
    # Selecionar feature para análise detalhada
    feature = st.selectbox("Selecione uma feature para análise detalhada:", 
                         df.columns[:-1])  # Excluindo a coluna Outcome
    
    # Criar subplots para análise da feature
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Box Plot", "Histograma", 
                                      "Box Plot por Classe", "Dispersão vs Glicose"))
    
    # Box Plot geral com outliers
    fig.add_trace(
        go.Box(y=df[feature], name=feature, boxpoints='outliers'),
        row=1, col=1
    )
    
    # Histograma com distribuição por classe
    fig.add_trace(
        go.Histogram(x=df[df['Outcome'] == 0][feature], name='Não Diabético', opacity=0.7),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=df[df['Outcome'] == 1][feature], name='Diabético', opacity=0.7),
        row=1, col=2
    )
    
    # Box Plot por classe com outliers
    fig.add_trace(
        go.Box(y=df[df['Outcome'] == 0][feature], name='Não Diabético', boxpoints='outliers'),
        row=2, col=1
    )
    fig.add_trace(
        go.Box(y=df[df['Outcome'] == 1][feature], name='Diabético', boxpoints='outliers'),
        row=2, col=1
    )
    
    # Dispersão vs Glicose com cores por classe
    fig.add_trace(
        go.Scatter(x=df['Glucose'], y=df[feature], 
                  mode='markers', 
                  marker=dict(color=df['Outcome'], 
                            colorscale='Viridis',
                            showscale=True,
                            size=8,
                            opacity=0.7)),
        row=2, col=2
    )
    
    fig.update_layout(height=800, width=1000, 
                     title_text=f"Análise Detalhada de {feature}",
                     showlegend=True)
    st.plotly_chart(fig)
    
    # 3. Estatísticas Descritivas
    st.subheader("Estatísticas Descritivas")
    
    # Estatísticas por classe
    stats_non_diabetic = df[df['Outcome'] == 0][feature].describe()
    stats_diabetic = df[df['Outcome'] == 1][feature].describe()
    
    stats_df = pd.DataFrame({
        'Não Diabético': stats_non_diabetic,
        'Diabético': stats_diabetic
    })
    
    st.dataframe(stats_df)

elif page == "Análise de Correlação":
    st.header("Análise de Correlação")
    
    # 1. Matriz de Correlação
    st.subheader("Matriz de Correlação")
    corr_matrix = df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matriz de Correlação entre Features",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig)
    
    # 2. Gráficos de Dispersão
    st.subheader("Gráficos de Dispersão")
    
    # Selecionar features para análise
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Selecione a feature X:", df.columns[:-1])
    with col2:
        y_feature = st.selectbox("Selecione a feature Y:", df.columns[:-1])
    
    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color='Outcome',
        title=f'Dispersão entre {x_feature} e {y_feature}',
        marginal_x='histogram',
        marginal_y='histogram'
    )
    st.plotly_chart(fig)

elif page == "Análise de Outliers":
    st.header("Análise de Outliers")
    
    # 1. Visão Geral dos Outliers
    st.subheader("Visão Geral dos Outliers")
    
    # Calcular outliers para cada feature
    outlier_counts = {}
    for column in df.columns[:-1]:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        outlier_counts[column] = len(outliers)
    
    # Gráfico de barras de outliers
    fig = px.bar(
        x=list(outlier_counts.keys()),
        y=list(outlier_counts.values()),
        title='Número de Outliers por Feature',
        labels={'x': 'Feature', 'y': 'Número de Outliers'}
    )
    st.plotly_chart(fig)
    
    # 2. Análise Detalhada por Feature
    st.subheader("Análise Detalhada por Feature")
    
    feature = st.selectbox("Selecione uma feature para análise de outliers:", 
                         df.columns[:-1])
    
    # Calcular limites para a feature selecionada
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identificar outliers
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    
    # Gráfico de box plot com outliers destacados
    fig = go.Figure()
    
    # Adicionar box plot
    fig.add_trace(go.Box(
        y=df[feature],
        name=feature,
        boxpoints='outliers',
        marker_color='rgb(7,40,89)',
        line_color='rgb(7,40,89)'
    ))
    
    # Adicionar linhas de limite
    fig.add_hline(y=lower_bound, line_dash="dash", line_color="red")
    fig.add_hline(y=upper_bound, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f'Box Plot de {feature} com Outliers',
        yaxis_title=feature
    )
    st.plotly_chart(fig)
    
    # Exibir estatísticas dos outliers
    st.write(f"**Limite Inferior:** {lower_bound:.2f}")
    st.write(f"**Limite Superior:** {upper_bound:.2f}")
    st.write(f"**Número de Outliers:** {len(outliers)}")
    st.write(f"**Percentual de Outliers:** {(len(outliers)/len(df))*100:.2f}%")

elif page == "Processo de Mineração":
    st.header("Processo de Mineração de Dados")
    
    # 1. Visão Geral do Processo
    st.subheader("Visão Geral do Processo")
    st.markdown("""
    O processo de mineração de dados para este projeto seguiu as seguintes etapas:
    
    1. **Coleta e Preparação dos Dados**
       - Carregamento do dataset Diabetes.csv
       - Verificação de valores ausentes
       - Análise inicial da distribuição dos dados
    
    2. **Análise Exploratória**
       - Estatísticas descritivas
       - Visualização da distribuição das classes
       - Identificação de outliers
       - Análise de correlações
    
    3. **Pré-processamento**
       - Tratamento de outliers usando o método IQR
       - Normalização dos dados
       - Balanceamento das classes usando SMOTE
    
    4. **Modelagem**
       - Treinamento de modelos (Random Forest e Gradient Boosting)
       - Validação cruzada
       - Avaliação de métricas de performance
    
    5. **Deploy e Visualização**
       - Criação da interface Streamlit
       - Visualização interativa dos resultados
       - Sistema de predição em tempo real
    """)
    
    # 2. Detalhes do Pré-processamento
    st.subheader("Detalhes do Pré-processamento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Tratamento de Outliers:**
        - Utilização do método IQR (Interquartile Range)
        - Substituição de outliers pelos limites inferior e superior
        - Preservação da distribuição original dos dados
        """)
    
    with col2:
        st.markdown("""
        **Balanceamento de Classes:**
        - Aplicação da técnica SMOTE
        - Geração de amostras sintéticas da classe minoritária
        - Balanceamento 50-50 entre as classes
        """)
    
    # 3. Visualização do Processo
    st.subheader("Visualização do Processo")
    
    # Gráfico mostrando a distribuição antes e depois do balanceamento
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Antes do Balanceamento", "Depois do Balanceamento"))
    
    # Distribuição antes
    fig.add_trace(
        go.Histogram(x=df['Outcome'], name='Original'),
        row=1, col=1
    )
    
    # Distribuição depois (simulada)
    balanced_dist = pd.Series([len(df)/2, len(df)/2], index=[0, 1])
    fig.add_trace(
        go.Histogram(x=balanced_dist.index, y=balanced_dist.values, name='Balanceado'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, width=800, title_text="Efeito do Balanceamento de Classes")
    st.plotly_chart(fig)

else:  # Métricas do Modelo
    st.header("Métricas do Modelo")
    
    # Métricas de performance
    st.subheader("Performance do Modelo")
    metrics = model_metrics.iloc[0]  # Assumindo que o melhor modelo é o primeiro
    
    # Criar métricas em colunas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Acurácia", f"{metrics['accuracy']*100:.2f}%")
    with col2:
        st.metric("Precisão", f"{metrics['precision']*100:.2f}%")
    with col3:
        st.metric("Recall", f"{metrics['recall']*100:.2f}%")
    with col4:
        st.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
    
    # Matriz de confusão
    st.subheader("Matriz de Confusão")
    conf_matrix = np.array([
        [metrics['true_negative'], metrics['false_positive']],
        [metrics['false_negative'], metrics['true_positive']]
    ])
    
    fig = px.imshow(
        conf_matrix,
        labels=dict(x="Predicted", y="Actual"),
        x=['Não Diabético', 'Diabético'],
        y=['Não Diabético', 'Diabético'],
        text_auto=True,
        aspect="auto",
        title="Matriz de Confusão"
    )
    st.plotly_chart(fig)
    
    # Métricas de validação cruzada
    st.subheader("Validação Cruzada")
    st.write(f"Média F1-Score (CV): {metrics['cv_f1_mean']*100:.2f}%")
    st.write(f"Desvio Padrão F1-Score (CV): {metrics['cv_f1_std']*100:.2f}%")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com ❤️ para predição de diabetes") 