import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.markdown("Preveja o preço de casas com Machine Learning")

st.markdown("---")

model = joblib.load("models/model.pkl")

st.sidebar.header("📊 Dados da casa")

medinc = st.sidebar.slider("💵 Renda média da região ($)", 0.0, 15.0, 5.0)
houseage = st.sidebar.slider("🏠 Idade média das casas (anos)", 1.0, 50.0, 25.0)
averooms = st.sidebar.slider("🛏️ Média de cômodos por casa", 1.0, 10.0, 5.0)
avebedrms = st.sidebar.slider("🛌 Média de quartos de dormir", 0.5, 5.0, 1.0)
population = st.sidebar.slider("👥 População da região", 0.0, 5000.0, 1000.0)
aveoccup = st.sidebar.slider("🏢 Pessoas por residência", 1.0, 10.0, 3.0)
latitude = st.sidebar.slider("🌍 Latitude", 32.0, 42.0, 37.0)
longitude = st.sidebar.slider("🌍 Longitude", -125.0, -114.0, -120.0)

col1, col2 = st.columns(2)

# Inputs
with col1:
    st.subheader("📥 Dados inseridos")
    input_data = pd.DataFrame({
        "MedInc": [medinc],
        "HouseAge": [houseage],
        "AveRooms": [averooms],
        "AveBedrms": [avebedrms],
        "Population": [population],
        "AveOccup": [aveoccup],
        "Latitude": [latitude],
        "Longitude": [longitude]
    })
    st.dataframe(input_data)

# Previsão de valor de casas
with col2:
    st.subheader("📈 Resultado")

    if st.button("🔮 Prever preço"):
        prediction = model.predict(input_data)[0]

        price_usd = prediction * 100000

        st.success(f"💰 Preço estimado: ${price_usd:,.2f}")

        # Feedback visual
        if price_usd > 300000:
            st.markdown("🏡 Região de alto padrão")
        elif price_usd > 150000:
            st.markdown("🏠 Região de valor médio")
        else:
            st.markdown("🏚️ Região de baixo custo")

        st.balloons()

st.markdown("---")

# Footer
st.markdown("Desenvolvido com ❤️ usando Machine Learning")