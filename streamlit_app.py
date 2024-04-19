import streamlit as st
import pandas as pd
import plost
import pickle
import sklearn
import shap
import matplotlib.pyplot as plt

st.title('Best Dash💖')

st.write('This is gonna be XAI dashboard')

# Row A
st.markdown('### Metrics')

vars = [['education', 'Уровень образования сотрудника'], ['gender', 'Пол сотрудника'], 
        ['no_of_trainings', 'Кол-во тренингов, которые прошел сотрудник за последний год'],
        ['avg_training_score', 'Средняя оценка за пройденные тренинги'], ['age', 'Возраст сотрудника'], 
        ['previous_year_rating', 'Рейтинг сотрудника за прошлый год'],
        ['length_of_service', 'Кол-во лет, которое сотрудник работает в компании'], ['awards_won', 'Кол-во выигранных наград'],
        ['department_', 'Отдел, в котором работает сотрудник']]
vars_df = pd.DataFrame(vars, columns=['Переменная', 'Описание'])

st.table(vars_df)

# Row B
df = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/X_test.csv')

modelGB = pickle.load(open('modelGB.pkl', 'rb'))
prediction = modelGB.predict(df)
prediction_proba = modelGB.predict_proba(df)

st.subheader('Prediction')
st.write(prediction_proba[10])

# Row C
explainer = shap.Explainer(modelGB)
shap_values = explainer.shap_values(df)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, df, plot_type='bar')
st.pyplot(bbox_inches='tight')
