
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# Carregar o arquivo de dados com caminho relativo
file_path = 'dados/adult.data'

# Definir os nomes das colunas, já que o arquivo 'adult.data' não tem cabeçalho
column_names = ['age', 'workclass', 'sample_weight', 'education', 'education_num', 'marital_status', 
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                'hours_per_week', 'native_country', 'income']

# Carregar os dados
dados = pd.read_csv(file_path, header=None, names=column_names, na_values=' ?', sep=',\s*', engine='python')

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

workclass = st.selectbox('Workclass:', ['Private', 'Self-emp-not-inc', 'Local-gov', '?', 'State-gov', 'Federal-gov', 'Without-pay', 'Never-worked'])
education = st.selectbox('Education:', ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
marital_status = st.selectbox('Estado Civil:', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox('Ocupação:', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox('Relacionamento:', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox('Raça:', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
sex = st.selectbox('Sexo:', ['Female', 'Male'])
native_country = st.selectbox('País:', ['United-States', 'Mexico', 'Greece', 'Vietnam', 'South', 'Puerto-Rico', 'Honduras', 'Japan', 'Canada', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Haiti', 'Taiwan', 'India', 'Ireland', 'Cambodia', 'Poland', 'England', 'China', 'Cuba', 'Jamaica', 'Iran', 'Italy', 'Philippines', 'Columbia', 'Thailand', 'Germany', 'Dominican-Republic', 'Laos', 'Ecuador', 'France', 'Nicaragua', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Hungary', 'Holand-Netherlands'])

# Botão de Previsão
if st.button("Prever Renda"):
    # Coletar os valores de entrada
    values = [age, workclass, sample_weight, education, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country]
    
    # Adicionando valor faltante para 'education_num'
    education_num_value = dados.loc[dados['education'] == education, 'education_num'].mean()

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
