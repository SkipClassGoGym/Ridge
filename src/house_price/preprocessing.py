"""Build the (unfitted) ColumnTransformer for house price features."""


import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = 'SalePrice'
DROP_COLS = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
LOG_COLS = ["GrLivArea", "GarageArea", "TotalBsmtSF", "1stFlrSF", "TotRmsAbvGrd", "YearBuilt"]

def _other_numerical_cols(X):
    result = [c for c in X.select_dtypes("number").columns if c not in LOG_COLS + DROP_COLS]
    return result

def _categorical_cols(X):
    result = [c for c in X.select_dtypes(include =['object', 'category']).columns if c not in  DROP_COLS]
    return result


def build_preprocessor():
    log_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler())
    ])

    num_pipe  = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value= "None")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("log", log_pipe ,LOG_COLS),
        ("num", num_pipe,_other_numerical_cols),
        ("cat",cat_pipe,_categorical_cols)
    ])
