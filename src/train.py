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
            "RMSE": rmse,
            "R2": r2
        }

    df_results = pd.DataFrame(results).T

    print("\n📊 Comparação de modelos:")
    print(df_results)

    return df_results


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

def plot_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        import matplotlib.pyplot as plt

        importances = model.feature_importances_
        features = X.columns

        plt.figure()
        plt.barh(features, importances)
        plt.xlabel("Importância")
        plt.title("Feature Importance (Random Forest)")
        plt.show()


def main():
    df = load_data()

    explore_data(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    models = train_models(X_train, y_train)

    evaluate_models(models, X_test, y_test)

    best_model = models["Random Forest"]

    y_pred = best_model.predict(X_test)
    plot_results(y_test, y_pred)

    save_model(best_model)

    plot_feature_importance(best_model, X_train)


if __name__ == "__main__":
    main()