from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd





def trein_model():
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
    return roc_auc, log_reg


def make_prediction(X_got):
    roc_auc, log_reg = trein_model
    prediction = log_reg.predict_proba(X_got)
    return prediction