import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Carregar os dados
df = pd.read_csv('Diabetes.csv')

# 1. Análise Descritiva e Tratamento de Outliers
print("\n=== Análise Descritiva e Tratamento de Outliers ===")

# Função para detectar e tratar outliers usando IQR
def treat_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identificar outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    # Substituir outliers pelos limites
    df.loc[df[column] < lower_bound, column] = lower_bound
    df.loc[df[column] > upper_bound, column] = upper_bound
    
    return df, len(outliers), lower_bound, upper_bound

# Tratar outliers para cada feature
outlier_stats = {}
for column in df.columns[:-1]:  # Excluindo a coluna target
    df, outlier_count, lower_bound, upper_bound = treat_outliers(df, column)
    outlier_stats[column] = {
        'outliers_count': outlier_count,
        'outliers_percentage': (outlier_count / len(df)) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

# Salvar estatísticas de outliers
outlier_stats_df = pd.DataFrame(outlier_stats).T
outlier_stats_df.to_csv('outlier_statistics.csv')

# 2. Análise Descritiva Detalhada
desc_stats = {}
for column in df.columns:
    stats = df[column].describe()
    if column != 'Outcome':
        # Adicionar estatísticas específicas para variáveis numéricas
        stats['skewness'] = df[column].skew()
        stats['kurtosis'] = df[column].kurtosis()
        stats['missing_values'] = df[column].isnull().sum()
        stats['missing_percentage'] = (df[column].isnull().sum() / len(df)) * 100
        stats['unique_values'] = df[column].nunique()
        
        # Estatísticas por classe (diabético vs não diabético)
        stats['mean_diabetic'] = df[df['Outcome'] == 1][column].mean()
        stats['mean_non_diabetic'] = df[df['Outcome'] == 0][column].mean()
        stats['std_diabetic'] = df[df['Outcome'] == 1][column].std()
        stats['std_non_diabetic'] = df[df['Outcome'] == 0][column].std()
    else:
        # Estatísticas específicas para a variável target
        stats['diabetic_percentage'] = (df[column].sum() / len(df)) * 100
        stats['non_diabetic_percentage'] = 100 - stats['diabetic_percentage']
    
    desc_stats[column] = stats

# Converter para DataFrame e salvar
desc_stats_df = pd.DataFrame(desc_stats).T
desc_stats_df.to_csv('descriptive_statistics.csv')

# Criar um dicionário com as descrições das features
feature_descriptions = {
    'Pregnancies': 'Number of times pregnant',
    'Glucose': 'Plasma glucose concentration (2 hours in an oral glucose tolerance test)',
    'BloodPressure': 'Diastolic blood pressure (mm Hg)',
    'SkinThickness': 'Triceps skin fold thickness (mm)',
    'Insulin': '2-Hour serum insulin (mu U/ml)',
    'BMI': 'Body mass index (weight in kg/(height in m)^2)',
    'DiabetesPedigreeFunction': 'Diabetes pedigree function',
    'Age': 'Age in years',
    'Outcome': 'Target variable (0 = No diabetes, 1 = Diabetes)'
}

# Análise detalhada de cada feature
print("\n=== Análise Detalhada das Features ===")
feature_analysis = {}
for column in df.columns[:-1]:  # Excluindo a coluna target
    stats_dict = {
        'description': feature_descriptions[column],
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'missing_values': df[column].isnull().sum()
    }
    
    # Criar visualização para cada feature
    plt.figure(figsize=(12, 4))
    
    # Subplot 1: Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    
    # Subplot 2: Distribuição por classe
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=column, hue='Outcome', multiple="stack")
    plt.title(f'Distribution of {column} by Outcome')
    
    plt.tight_layout()
    plt.savefig(f'feature_analysis_{column}.png')
    plt.close()

# Salvar análise detalhada
feature_analysis_df = pd.DataFrame(feature_analysis).T
feature_analysis_df.to_csv('feature_analysis.csv')

# 3. Preparação dos dados
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split dos dados
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Balanceamento das classes usando SMOTE
print("\n=== Balanceamento das Classes ===")
print(f"Distribuição antes do balanceamento: {pd.Series(y_train).value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Distribuição após o balanceamento: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# 5. Treinamento e comparação dos modelos
models = {
    'Random Forest': RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=10,
        class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=5
    )
}

results = {}
best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"\n=== Treinando {name} ===")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=10, scoring='f1')
    
    # Treinamento
    model.fit(X_train_balanced, y_train_balanced)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance_df.to_csv(f'feature_importance_{name.replace(" ", "_")}.csv', index=False)
    
    # Avaliação
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'true_negative': cm[0, 0],
        'false_positive': cm[0, 1],
        'false_negative': cm[1, 0],
        'true_positive': cm[1, 1]
    }
    
    # Atualizar melhor modelo
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# Salvar métricas dos modelos
results_df = pd.DataFrame(results).T
results_df.to_csv('model_metrics.csv')

# Salvar o melhor modelo, scaler e smote
joblib.dump(best_model, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(smote, 'smote.joblib')

print("\n=== Melhor Modelo ===")
print(f"Modelo: {best_model.__class__.__name__}")
print(f"F1-Score: {best_f1:.4f}")

# 7. Criar um oráculo para previsão
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    new_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Tratar outliers nos novos dados
    for column in new_data.columns:
        lower_bound = outlier_stats[column]['lower_bound']
        upper_bound = outlier_stats[column]['upper_bound']
        new_data[column] = new_data[column].clip(lower=lower_bound, upper=upper_bound)
    
    # Normalizar os dados
    new_data_scaled = scaler.transform(new_data)
    
    # Fazer a previsão
    prediction = best_model.predict(new_data_scaled)
    probability = best_model.predict_proba(new_data_scaled)
    
    return prediction[0], probability[0]

# Exemplo de uso do oráculo
print("\n=== Exemplo de Previsão ===")
pred, prob = predict_diabetes(6, 148, 72, 35, 0, 33.6, 0.627, 50)
print(f"Previsão: {'Diabético' if pred == 1 else 'Não Diabético'}")
print(f"Probabilidade: {prob[1]:.2%} de ser diabético")

# 9. Gerar exemplos de previsão para Power BI
sample_predictions = []
for _ in range(10):
    sample = df.sample(1)
    pred, prob = predict_diabetes(
        sample['Pregnancies'].values[0],
        sample['Glucose'].values[0],
        sample['BloodPressure'].values[0],
        sample['SkinThickness'].values[0],
        sample['Insulin'].values[0],
        sample['BMI'].values[0],
        sample['DiabetesPedigreeFunction'].values[0],
        sample['Age'].values[0]
    )
    sample_predictions.append({
        'Pregnancies': sample['Pregnancies'].values[0],
        'Glucose': sample['Glucose'].values[0],
        'BloodPressure': sample['BloodPressure'].values[0],
        'SkinThickness': sample['SkinThickness'].values[0],
        'Insulin': sample['Insulin'].values[0],
        'BMI': sample['BMI'].values[0],
        'DiabetesPedigreeFunction': sample['DiabetesPedigreeFunction'].values[0],
        'Age': sample['Age'].values[0],
        'Prediction': pred,
        'Probability_Diabetes': prob[1]
    })

pd.DataFrame(sample_predictions).to_csv('sample_predictions.csv', index=False)

print("\n=== Processo Concluído ===")
print("Arquivos gerados para Power BI:")
print("1. descriptive_statistics.csv - Estatísticas descritivas")
print("2. outlier_statistics.csv - Estatísticas de outliers")
print("3. model_metrics.csv - Métricas do modelo")
print("4. sample_predictions.csv - Exemplos de previsões")
print("\nModelo salvo em best_model.joblib")
print("Scaler salvo em scaler.joblib")
print("SMOTE salvo em smote.joblib")

# 10. Gerar visualizações adicionais
# Matriz de correlação
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Distribuição da variável target
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Outcome')
plt.title('Distribution of Diabetes Cases')
plt.savefig('target_distribution.png')
plt.close() 