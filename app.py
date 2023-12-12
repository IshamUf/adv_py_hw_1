import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ready_file.csv')


df[['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER']] = \
    df[['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER']].astype(str)

st.title("Разведочный анализ данных")

st.subheader("Ниже вы можете увидеть первые пять строк подготовленной таблицы для анализа")

st.dataframe(df.head())


st.header('Графики распределения признаков')
column_distribution = ['AGE','CHILD_TOTAL','DEPENDANTS','PERSONAL_INCOME','FAMILY_INCOME']
for column in column_distribution:
    fig = px.histogram(df, x=column, nbins=20)
    fig.update_layout(title_text=f'Распределение {column}')
    fig

fig = px.histogram(df, x='FAMILY_INCOME', nbins=20)
fig.update_layout(title_text=f'Распределение FAMILY_INCOME')
fig.update_xaxes(tickangle=45)
fig


from plotly.subplots import make_subplots


data_gender = df.groupby(['TARGET', 'GENDER']).size().reset_index(name='Count')
data_work = df.groupby(['TARGET', 'SOCSTATUS_WORK_FL']).size().reset_index(name='Count')
data_pens = df.groupby(['TARGET', 'SOCSTATUS_PENS_FL']).size().reset_index(name='Count')
fig = make_subplots(rows=1, cols=3, subplot_titles=[
    'Пол 1-муж, 0-жен',
    'Статус работы 1-раб, 0-не раб',
    'Статус Пенсии 1-пенс, 0-не пенс'
])
fig.add_trace(px.bar(data_gender, x='GENDER', y='Count', color='TARGET').data[0], row=1, col=1)
fig.add_trace(px.bar(data_work, x='SOCSTATUS_WORK_FL', y='Count', color='TARGET').data[0], row=1, col=2)
fig.add_trace(px.bar(data_pens, x='SOCSTATUS_PENS_FL', y='Count', color='TARGET').data[0], row=1, col=3)
fig.update_layout(title='Категориальные признаки: Пол, Статус работы, Статус Пенсии')
fig


st.header('Матрица корреляций')
corr_matrix = df[df.select_dtypes(include=['int64', 'float64']).columns].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
st.pyplot(fig)


st.header('Графики зависимости целевой переменной и признаков')
fig = px.scatter(df, x='TARGET', y='PERSONAL_INCOME', title='Scatter Plot of PERSONAL_INCOME vs TARGET')
fig

fig = px.scatter(df, x='TARGET', y='CREDIT', title='Scatter Plot of CREDIT vs TARGET')
fig




st.header('Числовые характеристики')
df_num = df.copy()
df_num.drop('AGREEMENT_RK', axis=1, inplace=True)
df_num.drop('ID_CLIENT', axis=1, inplace=True)
stats = df_num.describe().transpose()
st.write(stats)

st.header('Давай предскажем')

AGE = st.slider("Введите AGE", min_value=int(df['AGE'].min()), max_value=int(df['AGE'].max()), value=20, step=1)
SOCSTATUS_WORK_FL = st.radio('Работаете? 1-Да: ', df['SOCSTATUS_WORK_FL'].unique())
SOCSTATUS_PENS_FL = st.radio('На пенсии? 1-Да: ', df['SOCSTATUS_PENS_FL'].unique())
GENDER = st.radio('1-муж, 0-жен: ', df['GENDER'].unique())
CHILD_TOTAL = st.radio('Сколько детей: ', df['CHILD_TOTAL'].unique())
DEPENDANTS = st.radio('Количество иждивенцев: ', df['DEPENDANTS'].unique())
PERSONAL_INCOME = st.slider("Введите PERSONAL_INCOME", min_value=int(df['PERSONAL_INCOME'].min()), max_value=int(df['PERSONAL_INCOME'].max()), value=20, step=1)
FAMILY_INCOME = st.radio('Укажите единицы измерения роста: ', df['FAMILY_INCOME'].unique())
LOAN_NUM_TOTAL = st.radio('Всего кредитов: ', df['LOAN_NUM_TOTAL'].unique())
LOAN_NUM_CLOSED = st.radio('Всего кредитов закрыто: ', df['LOAN_NUM_CLOSED'].unique())

GEN_INDUSTRY = st.radio('INDUSTRY: ', df['GEN_INDUSTRY'].unique())
GEN_TITLE = st.radio('TITLE: ', df['GEN_TITLE'].unique())
JOB_DIR = st.radio('DIR: ', df['JOB_DIR'].unique())

WORK_TIME = st.slider("Введите WORK_TIME", min_value=int(df['WORK_TIME'].min()), max_value=int(df['WORK_TIME'].max()), value=20, step=1)

CREDIT = st.slider("Введите CREDIT", min_value=int(df['CREDIT'].min()), max_value=int(df['CREDIT'].max()), value=20, step=1)
TERM = st.slider("Введите TERM", min_value=int(df['TERM'].min()), max_value=int(df['TERM'].max()), value=20, step=1)
FST_PAYMENT = st.slider("Введите FST_PAYMENT", min_value=int(df['FST_PAYMENT'].min()), max_value=int(df['FST_PAYMENT'].max()), value=20, step=1)




var1 = 10
var2 = 'Hello'
var3 = [1, 2, 3]

# Создание DataFrame из переменных
data = {'Variable_Name': ['AGE', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME',
                          'FAMILY_INCOME', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED', 'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR',
                          'WORK_TIME', 'GEN_INDUSTRY_was_null' , 'GEN_TITLE_was_null' , 'JOB_DIR_was_null', 'WORK_TIME_was_null'
                          'CREDIT', 'TERM', 'FST_PAYMENT'],
        'Variable_Value': [int(AGE), str(SOCSTATUS_WORK_FL), str(SOCSTATUS_PENS_FL), str(GENDER), int(CHILD_TOTAL), int(DEPENDANTS), float(PERSONAL_INCOME),
                          str(FAMILY_INCOME), int(LOAN_NUM_TOTAL), int(LOAN_NUM_CLOSED), str(GEN_INDUSTRY), str(GEN_TITLE), str(JOB_DIR),
                          float(WORK_TIME),0,0,0,0,
                           float(CREDIT), int(TERM), float(FST_PAYMENT)]}

df_got = pd.DataFrame(data)

predicted = predict(df_got)