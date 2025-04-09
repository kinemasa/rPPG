from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

def get_model(model_name, params, rand=0):
    """
    モデル名に応じた回帰モデルのインスタンスを返す

    Parameters
    ----------
    model_name : str
        使用するモデル名（"rf", "svr", "lgbm", "gbr", "ridge"）
    params : dict
        モデルに渡すパラメータ
    rand : int
        random_stateに使う固定値（モデルによっては不要）

    Returns
    -------
    sklearnのモデルインスタンス
    """
    if model_name == "rf":
        return RandomForestRegressor(**params)
    elif model_name == "svr":
        return SVR(**params)
    elif model_name == "lgbm":
        return LGBMRegressor(**params)
    elif model_name == "gbr":
        return GradientBoostingRegressor(**params)
    elif model_name == "ridge":
        return Ridge(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")