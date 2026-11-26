"""Building pipeline model - Ridge """


import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_model(alpha =1.0):
    model_pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model",TransformedTargetRegressor(Ridge(alpha = alpha), func= np.log1p, inverse_func=np.expm1))
    ])

    return model_pipe