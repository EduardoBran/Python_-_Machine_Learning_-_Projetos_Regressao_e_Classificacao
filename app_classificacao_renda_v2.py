
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier

# Carregar o arquivo de dados com caminho relativo
file_path = 'dados/adult.data'

# Definir os nomes das colunas, já que o arquivo 'adult.data' não tem cabeçalho
column_names = ['age', 'workclass', 'sample_weight', 'education', 'education_num', 'marital_status', 
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                'hours_per_week', 'native_country', 'income']

# Carregar os dados
dados = pd.read_csv(file_path, header=None, names=column_names, na_values=' ?', sep=',\s*', engine='python')

# Codificar variáveis categóricas
categorical_vars = ['workclass', 'education', 'marital_status', 'occupation', 
                    'relationship', 'race', 'sex', 'native_country']

label_encoders = {}
for var in categorical_vars:
    le = LabelEncoder()
    dados[var] = le.fit_transform(dados[var].astype(str))
    label_encoders[var] = le

# Carregar o modelo treinado
modelo_CatBoost = CatBoostClassifier()
modelo_CatBoost.load_model('modelos_imagens/modelo_CatBoost.cbm')

# Carregar os dados para padronização
variaveis_continuas = ['age', 'sample_weight', 'capital_gain', 'capital_loss', 'hours_per_week']
scaler = StandardScaler()
scaler.fit(dados[variaveis_continuas])

# Ordem das colunas esperada pelo modelo
ordem_colunas_modelo = [
    'age', 'workclass', 'sample_weight', 'education', 'education_num', 'marital_status', 
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
    'hours_per_week', 'native_country'
]

# Título da Aplicação
st.title("Previsão de Renda com CatBoost")

# Campos de entrada para os valores das variáveis
age = st.number_input('Idade:', min_value=17, max_value=90)
sample_weight = st.number_input('Peso Amostral:', min_value=12285, max_value=1484705)
capital_gain = st.number_input('Ganho de Capital:', min_value=0, max_value=99999)
capital_loss = st.number_input('Perda de Capital:', min_value=0, max_value=4356)
hours_per_week = st.number_input('Horas/Semana:', min_value=1, max_value=99)

workclass = st.selectbox('Workclass:', label_encoders['workclass'].classes_)
education = st.selectbox('Education:', label_encoders['education'].classes_)
marital_status = st.selectbox('Estado Civil:', label_encoders['marital_status'].classes_)
occupation = st.selectbox('Ocupação:', label_encoders['occupation'].classes_)
relationship = st.selectbox('Relacionamento:', label_encoders['relationship'].classes_)
race = st.selectbox('Raça:', label_encoders['race'].classes_)
sex = st.selectbox('Sexo:', label_encoders['sex'].classes_)
native_country = st.selectbox('País:', label_encoders['native_country'].classes_)

# Botão de Previsão
if st.button("Prever Renda"):
    # Coletar os valores de entrada
    values = [
        age,
        label_encoders['workclass'].transform([workclass])[0],
        sample_weight,
        label_encoders['education'].transform([education])[0],
        label_encoders['marital_status'].transform([marital_status])[0],
        label_encoders['occupation'].transform([occupation])[0],
        label_encoders['relationship'].transform([relationship])[0],
        label_encoders['race'].transform([race])[0],
        label_encoders['sex'].transform([sex])[0],
        capital_gain,
        capital_loss,
        hours_per_week,
        label_encoders['native_country'].transform([native_country])[0]
    ]
    
    # Adicionando valor faltante para 'education_num'
    education_num_value = dados.loc[dados['education'] == label_encoders['education'].transform([education])[0], 'education_num'].mean()

    # Criação do DataFrame com a ordem correta das colunas
    values.insert(4, education_num_value)
    data = pd.DataFrame([values], columns=[
        'age', 'workclass', 'sample_weight', 'education', 'education_num', 'marital_status', 
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
        'hours_per_week', 'native_country'])
    
    # Verificação da ordem das colunas
    data = data[ordem_colunas_modelo]
    
    # Padronização das variáveis contínuas
    data[variaveis_continuas] = scaler.transform(data[variaveis_continuas])
    
    # Previsão
    prediction = modelo_CatBoost.predict(data)
    proba = modelo_CatBoost.predict_proba(data)[0]
    
    # Exibição do resultado
    result = " >50K" if prediction[0] == 1 else " <=50K"
    confidence = proba[prediction[0]] * 100
    
    st.write(f"**Este modelo calculou que há {confidence:.1f}% de chance da renda ser {result}.**")
