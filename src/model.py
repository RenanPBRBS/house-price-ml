from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_models(X_train, y_train):
    models = {}

    # Regressão Linear com normalização
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    models["Linear Regression"] = lr_pipeline

    # Random Forest (não precisa de scaler mas ok se manter padrão)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    return models