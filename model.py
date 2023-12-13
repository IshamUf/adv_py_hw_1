from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

df_model = pd.read_csv('ready_file.csv')



def make_prediction_logic_reg(X_got):
    df_model[['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER']] = \
        df_model[['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER']].astype(str)
    y = df_model['TARGET']
    X = df_model.copy()
    X.drop(columns=['AGREEMENT_RK','ID_CLIENT','TARGET'], inplace=True)
    X = pd.get_dummies(X)
    X_got = pd.get_dummies(X_got)

    for i in set(X.columns) - set(X_got.columns):
        X_got[i] = False
    X_got = X_got[X.columns]

    X.drop(columns=['SOCSTATUS_WORK_FL_0','SOCSTATUS_PENS_FL_0','GENDER_0','FAMILY_INCOME_до 5000 руб.','GEN_INDUSTRY_Банк/Финансы'
        ,'GEN_TITLE_Военнослужащий по контракту','JOB_DIR_Адм-хоз. и трансп. службы'], inplace=True)
    X_got.drop(columns=['SOCSTATUS_WORK_FL_0','SOCSTATUS_PENS_FL_0','GENDER_0','FAMILY_INCOME_до 5000 руб.','GEN_INDUSTRY_Банк/Финансы'
        ,'GEN_TITLE_Военнослужащий по контракту','JOB_DIR_Адм-хоз. и трансп. службы'], inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    X_got = scaler.transform(X_got)
    roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
    prediction = log_reg.predict_proba(X_got)

    return prediction,roc_auc