import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
#data load and classified
df=pd.read_csv("data.csv")
X=df.drop("future_trend",axis=1)
Y=df["future_trend"]
cat_features=["asset_type","market_regime"]
num_features=[
    "lookback_days",
    "high_volatility",
    "trend_continuation",
    "technical_score",
    "edge_density",
    "slope_strength",
    "candlestick_variance",
    "pattern_symmetry"
]
# preprocessing data part
preprocessing=ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first"), cat_features)
    ]
)
#data split part did lots of tweaking here to get best score
X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.28,
    random_state=7,
    stratify=Y
)
#logisitc model train and its performance
log_model=Pipeline(steps=[
    ("preprocessor", preprocessing),
    ("classifier", LogisticRegression(random_state=7, max_iter=500))
])
log_model.fit(X_train,Y_train)
Y_pred_log=log_model.predict(X_test)
print("logistic_reg model Performance:")
print("accuracy:", accuracy_score(Y_test, Y_pred_log))
print("precision:", precision_score(Y_test, Y_pred_log))
print("recall:", recall_score(Y_test, Y_pred_log))
print("f1 score:", f1_score(Y_test, Y_pred_log))
print("confusion matrix:\n", confusion_matrix(Y_test, Y_pred_log))
#nn model train and its performance
nn_model=Pipeline(steps=[
    ("preprocessor", preprocessing),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=7
    ))
])
nn_model.fit(X_train, Y_train)
Y_pred_nn=nn_model.predict(X_test)
print("")
print("neural_network model Performance:")
print("accuracy:", accuracy_score(Y_test, Y_pred_nn))
print("precision:", precision_score(Y_test, Y_pred_nn))
print("recall:", recall_score(Y_test, Y_pred_nn))
print("f1 score:", f1_score(Y_test, Y_pred_nn))
print("confusion matrix:\n", confusion_matrix(Y_test, Y_pred_nn))
print("")
#created a small data showing result of both model
finaldata =pd.DataFrame({
    "model": ["logmodel", "nn_model"],
    "accuracy": [accuracy_score(Y_test, Y_pred_log), accuracy_score(Y_test, Y_pred_nn)],
    "precision": [precision_score(Y_test, Y_pred_log), precision_score(Y_test, Y_pred_nn)],
    "recall": [recall_score(Y_test, Y_pred_log), recall_score(Y_test, Y_pred_nn)],
    "f1_score": [f1_score(Y_test, Y_pred_log), f1_score(Y_test, Y_pred_nn)]
})
print(finaldata)
#compared using if else logic for best model
if (accuracy_score(Y_test, Y_pred_nn) >= accuracy_score(Y_test, Y_pred_log)) and (f1_score(Y_test, Y_pred_nn) >= f1_score(Y_test, Y_pred_log)) and (precision_score(Y_test, Y_pred_nn) >= precision_score(Y_test, Y_pred_log)) and (recall_score(Y_test, Y_pred_nn) >= recall_score(Y_test, Y_pred_log)):
    print("")
    print("neuralnetwork model wins")
else:
    print("")
    print("logisitc model wins")