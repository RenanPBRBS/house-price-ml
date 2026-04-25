from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_models(X_train, y_train):
    models = {}

    # Regressão Linear
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = lr

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    return models