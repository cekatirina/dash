import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plost
import pickle
import sklearn
import shap
import matplotlib.pyplot as plt

st.title('Best Dash💖')

st.write('This is gonna be XAI dashboard')

# Row A
vars = [['education', 'Уровень образования сотрудника'], ['gender', 'Пол сотрудника'], 
        ['no_of_trainings', 'Кол-во тренингов, которые прошел сотрудник за последний год'],
        ['avg_training_score', 'Средняя оценка за пройденные тренинги'], ['age', 'Возраст сотрудника'], 
        ['previous_year_rating', 'Рейтинг сотрудника за прошлый год'],
        ['length_of_service', 'Кол-во лет, которое сотрудник работает в компании'], ['awards_won', 'Кол-во выигранных наград'],
        ['department_', 'Отдел, в котором работает сотрудник']]
vars_df = pd.DataFrame(vars, columns=['Переменная', 'Описание'])

explainer = shap.Explainer(modelGB)
shap_values = explainer.shap_values(df)

st.markdown('### Metrics')
col1, col2 = st.columns(2)
col1.table(vars_df)

plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, df, plot_type='bar')
col2.pyplot(bbox_inches='tight')

# Row B
df = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/X_test.csv')

modelGB = pickle.load(open('modelGB.pkl', 'rb'))
prediction = modelGB.predict(df)
prediction_proba = modelGB.predict_proba(df)

st.subheader('Prediction')
st.write(prediction_proba[10])
