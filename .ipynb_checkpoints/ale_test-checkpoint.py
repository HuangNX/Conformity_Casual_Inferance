import numpy as np
import pandas as pd
from loguru import logger
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import matplotlib as mpl
from ale import ale_plot

def model_predict(X, model_name):
    return model.predict(X)

if __name__ == "__main__":
    iris = datasets.load_iris()
    dataset = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]

    model = RandomForestRegressor(n_estimators=20, bootstrap=True)
    model.fit(X, y)
    model_name='test'
    predictor = lambda x: model_predict(x, model_name)

    mpl.rc("figure", figsize=(9, 6))
    ale_fig = ale_plot(
        None,
        X,
        ['sepal width (cm)'],
        bins=20,
        predictor=predictor,
        monte_carlo=True,
        monte_carlo_rep=100,
        monte_carlo_ratio=0.6,
    )

    ale_fig.savefig("ale_test.png")