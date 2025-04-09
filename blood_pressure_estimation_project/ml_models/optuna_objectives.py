from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

def objective_rf(trial, X, y):
    """
    RandomForestRegressor の Optuna 目的関数
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }

    model = RandomForestRegressor(**params)
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    return -score.mean()


def objective_svr(trial, X, y):
    """
    SVR の Optuna 目的関数
    """
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    epsilon = trial.suggest_loguniform("epsilon", 1e-3, 1e1)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3

    model = SVR(C=C, epsilon=epsilon, kernel=kernel, degree=degree)
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    return -score.mean()


def objective_lgbm(trial, X, y):
    """
    LightGBM の Optuna 目的関数
    """
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1.0),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7)
    }

    model = LGBMRegressor(**params)
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    return -score.mean()