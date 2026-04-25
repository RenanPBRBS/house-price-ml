import joblib
import pandas as pd

# carregar modelo
model = joblib.load("models/model.pkl")

# exemplo de entrada (mesmo formato do dataset)
data = {
    "MedInc": [8.0],
    "HouseAge": [40],
    "AveRooms": [6],
    "AveBedrms": [1],
    "Population": [300],
    "AveOccup": [3],
    "Latitude": [37.8],
    "Longitude": [-122.2]
}

df = pd.DataFrame(data)

# previsão
prediction = model.predict(df)

print(f"\n🏠 Preço previsto: {prediction[0]:.3f}")