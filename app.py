import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
page = st.sidebar.radio("Selecione uma página:", ["Predição", "Análise de Dados", "Métricas do Modelo"])

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
    fig = px.pie(
        values=class_dist.values,
        names=['Não Diabético', 'Diabético'],
        title='Distribuição de Pacientes Diabéticos vs Não Diabéticos'
    )
    st.plotly_chart(fig)
    
    # Estatísticas da distribuição
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Pacientes", len(df))
    with col2:
        st.metric("Proporção Diabéticos", f"{class_dist[1]/len(df)*100:.1f}%")
    
    # 2. Análise de Features
    st.subheader("Análise de Features")
    
    # Selecionar feature para análise detalhada
    feature = st.selectbox("Selecione uma feature para análise detalhada:", 
                         df.columns[:-1])  # Excluindo a coluna Outcome
    
    # Criar subplots para análise da feature
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Box Plot", "Histograma", 
                                      "Box Plot por Classe", "Dispersão vs Glicose"))
    
    # Box Plot geral
    fig.add_trace(
        go.Box(y=df[feature], name=feature),
        row=1, col=1
    )
    
    # Histograma
    fig.add_trace(
        go.Histogram(x=df[feature], name=feature),
        row=1, col=2
    )
    
    # Box Plot por classe
    fig.add_trace(
        go.Box(y=df[df['Outcome'] == 0][feature], name='Não Diabético'),
        row=2, col=1
    )
    fig.add_trace(
        go.Box(y=df[df['Outcome'] == 1][feature], name='Diabético'),
        row=2, col=1
    )
    
    # Dispersão vs Glicose
    fig.add_trace(
        go.Scatter(x=df['Glucose'], y=df[feature], 
                  mode='markers', 
                  marker=dict(color=df['Outcome'], 
                            colorscale='Viridis',
                            showscale=True)),
        row=2, col=2
    )
    
    fig.update_layout(height=800, width=1000, 
                     title_text=f"Análise Detalhada de {feature}")
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
    
    # 4. Análise de Correlação
    st.subheader("Análise de Correlação")
    
    # Matriz de correlação
    corr_matrix = df.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matriz de Correlação entre Features"
    )
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