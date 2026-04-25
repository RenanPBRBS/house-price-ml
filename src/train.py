import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch
from typing import cast
from model import train_models


def load_data():
    # Corrige problema de tipagem do Pylance
    data = cast(Bunch, fetch_california_housing(as_frame=True))
    df = data.frame
    return df


def explore_data(df):
    print("\n📊 Primeiras linhas:")
    print(df.head())

    print("\n📋 Informações:")
    print(df.info())

    print("\n📈 Estatísticas:")
    print(df.describe())


def prepare_data(df):
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)

        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        print(f"\n🔍 {name}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

    return results


def plot_results(y_test, y_pred):
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Valores reais")
    plt.ylabel("Previsões")
    plt.title("Real vs Predito")
    plt.show()


def save_model(model):
    joblib.dump(model, "models/model.pkl")
    print("\n✅ Modelo salvo em models/model.pkl")


def main():
    # 1. Carregar dados
    df = load_data()

    # 2. Explorar dados
    explore_data(df)

    # 3. Preparar dados
    X_train, X_test, y_train, y_test = prepare_data(df)

    # 4. Treinar modelos
    models = train_models(X_train, y_train)

    # 5. Avaliar modelos
    evaluate_models(models, X_test, y_test)

    # 6. Escolher melhor modelo (Random Forest)
    best_model = models["Random Forest"]

    # 7. Plot
    y_pred = best_model.predict(X_test)
    plot_results(y_test, y_pred)

    # 8. Salvar modelo
    save_model(best_model)


if __name__ == "__main__":
    main()