
import streamlit as st
from PIL import Image
import numpy as np
import io
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Função para pré-processamento da imagem
def preprocess_image(image_content, target_size=(128, 128)):
    img = Image.open(io.BytesIO(image_content))
    img_resized = img.resize(target_size)
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# Título da Aplicação
st.title("Classificação de Imagens de Flores")

# Carregar a ordem dos rótulos
with open('modelos_imagens/label_encoder_classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Upload de Imagem
uploaded_file = st.file_uploader("Escolha uma imagem de flor", type=["jpg", "jpeg", "png"])

# Botão de Classificação (Desabilitado Inicialmente)
if uploaded_file is not None:
    classify_button = st.button("Classificar")
    
    if classify_button:
        # Carregar a imagem
        img_array, img = preprocess_image(uploaded_file.read())
        
        # Carregar o modelo
        model = load_model('modelos_imagens/flower_classifier_model.h5')
        
        # Realizar a classificação
        prediction = model.predict(img_array)
        confidence = np.max(prediction) * 100
        result = classes[np.argmax(prediction)]
        
        # Exibir a imagem e a classificação
        st.image(img, caption='Imagem carregada', use_column_width=True)
        st.write(f"**Este modelo calculou que tem {confidence:.1f}% de chance dessa flor ser {result}.**")
