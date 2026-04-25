import streamlit as st
import pandas as pd
import joblib

st.markdown("---")

# Configuração da página
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 Previsão de Preço de Casas")
st.write("Preencha os dados abaixo para prever o valor de uma casa.")

# Carregar modelo
model = joblib.load("models/model.pkl")

# Inputs do usuário
medinc = st.number_input("Renda média (MedInc)", value=5.0)
houseage = st.number_input("Idade da casa (HouseAge)", value=30.0)
averooms = st.number_input("Média de quartos (AveRooms)", value=5.0)
avebedrms = st.number_input("Média de quartos de dormir (AveBedrms)", value=1.0)
population = st.number_input("População", value=1000.0)
aveoccup = st.number_input("Ocupação média (AveOccup)", value=3.0)
latitude = st.number_input("Latitude", value=37.0)
longitude = st.number_input("Longitude", value=-122.0)

# Botão de previsão
if st.button("🔮 Prever preço"):
    data = pd.DataFrame({
        "MedInc": [medinc],
        "HouseAge": [houseage],
        "AveRooms": [averooms],
        "AveBedrms": [avebedrms],
        "Population": [population],
        "AveOccup": [aveoccup],
        "Latitude": [latitude],
        "Longitude": [longitude]
    })

    prediction = model.predict(data)

    st.success(f"💰 Preço previsto: {prediction[0]:.2f}")

    st.balloons()