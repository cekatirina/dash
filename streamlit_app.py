import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plost
import pickle
import sklearn
import shap
import matplotlib.pyplot as plt
from shapash.explainer.smart_explainer import SmartExplainer

df = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/X_test.csv')
modelGB = pickle.load(open('modelGB.pkl', 'rb'))
prediction = modelGB.predict(df)
prediction_proba = modelGB.predict_proba(df)
explainer = shap.Explainer(modelGB)
shap_values = explainer.shap_values(df)

tab1, tab2 = st.tabs(["Дэшборд", "Анкета"])

with tab1:
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
        col1, col2 = st.columns(2)
        with col1:
                st.markdown('### Описание переменных')
                st.table(vars_df)
        with col2:
                st.markdown('### Важность предикторов')
                plt.title('Feature importance based on SHAP values')
                shap.summary_plot(shap_values, df, plot_type='bar')
                st.pyplot()
                
        # Row B
        response_dict = {0: 'Not promoted', 1:' Promoted'}
        xpl = SmartExplainer(model = modelGB,
                             label_dict=response_dict) # Optional parameters, dicts specify labels
        xpl.compile(x=df)
        st.subheader('Shapash')
        xpl.plot.contribution_plot(col='avg_training_score', max_points=9276)
        st.pyplot()
        
        # Row C
        st.subheader('Prediction')
        st.write(prediction_proba[10])

with tab2:
        st.title('Best Dash💖')
