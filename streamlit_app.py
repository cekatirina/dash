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
df_prob = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/X_test_prob.csv')
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
                ['department_', 'Отдел, в котором работает сотрудник'], ['recruitment_channel_', 'Как человек попал в компанию (referred - через реферальную программу, sourcing - обычный поиск сотрудников)']]
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
        st.markdown('### Важность предикторов')
        plt.title('Feature contribution based on SHAP values')
        shap.dependence_plot("avg_training_score", shap_values, df,
                    feature_names=df.columns, interaction_index="gender")
        st.pyplot()
        
        # Row C
        st.subheader('Prediction')
        st.write(prediction_proba[10])

with tab2:
        st.title('Best Dash💖')
        ANSWER_OPTIONS = [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]
        with st.form(key="dash_form"):
            name = st.text_input(label="ФИО")
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)    
            dificulty = st.radio(
                    "Насколько сложно было...",
                    ["1", "2", "3", "4", "5"])        
            submit_button = st.form_submit_button(label="Отправить анкету")
