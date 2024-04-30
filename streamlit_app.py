import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plost
import pickle
import sklearn
import shap
import matplotlib.pyplot as plt
from shapash.explainer.smart_explainer import SmartExplainer

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

df = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/X_test.csv')
y = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/y_test.csv')
df_prob = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/X_test_prob.csv')
modelGB = pickle.load(open('modelGB.pkl', 'rb'))
prediction = modelGB.predict(df)
prediction_proba = modelGB.predict_proba(df)
explainer = shap.Explainer(modelGB)
shap_values = explainer.shap_values(df)

example1 = df.iloc[1432]
example1 = example1.to_numpy()

example2 = df.iloc[3842]
example2 = example2.to_numpy()

tab1, tab2 = st.tabs(["Дэшборд", "Анкета"])

with tab1:
        st.title('Общая статистика по модели')
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
                st.markdown('### Важность переменных')
                plt.title('Feature importance based on SHAP values')
                plt.xlabel("Среднее SHAP value")
                shap.summary_plot(shap_values, df, plot_type='bar')
                st.pyplot()
        
        # Row B
        st.markdown('### Важность предикторов')
        plt.title('Feature contribution based on SHAP values')
        shap.dependence_plot("avg_training_score", shap_values, df, feature_names=df.columns, interaction_index="gender")
        st.pyplot()

        # Row C
        st.title('Информация по индивидуальным предсказаниям')
        c1, c2 = st.columns(2)
        with c1:
                st.markdown('### Пример 1 (ID = 1432)')
                shap.waterfall_plot(shap.Explanation(values=shap_values[1432],
                            base_values=explainer.expected_value[0],
                            data=example1,
                            feature_names=df.columns))
                st.pyplot()
        with c2:
                st.markdown('### Пример 2 (ID = 3842)')
                shap.waterfall_plot(shap.Explanation(values=shap_values[3842],
                            base_values=explainer.expected_value[0],
                            data=example2,
                            feature_names=df.columns))
                st.pyplot()

with tab2:
        st.title('Ваши впечатления от дэшборда')
        ANSWER_OPTIONS = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7"
        ]
        with st.form(key="dash_form"):
            st.markdown('Вам будет представлен ряд утвержений по поводу доверия модели, объяснение который вы изучили на дэшборде.  ')
            st.markdown('**Оцените каждое утверждение по шкале от 1 до 7** (1 - полностью НЕ согласен, 7 - полностью согласен)')
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)    
            trust1 = st.radio("Модель обманчива", ANSWER_OPTIONS) 
            trust2 = st.radio("Модель принимает решения нечестно", ANSWER_OPTIONS)
            trust3 = st.radio("Намерения модели вызывают у меня подозрения", ANSWER_OPTIONS)
            trust4 = st.radio("Я был бы осторожен по отношению к решениям, принимаемым этой моделью", ANSWER_OPTIONS)
            trust5 = st.radio("Результаты модели могут привести к вредоносным последствиям", ANSWER_OPTIONS)
            trust6 = st.radio("Я уверен в результатах модели", ANSWER_OPTIONS)
            trust7 = st.radio("Результаты модели безопасны для применения", ANSWER_OPTIONS)
            trust8 = st.radio("На результаты модели можно с уверенностью положиться", ANSWER_OPTIONS)
            trust9 = st.radio("Я доверяю результатам модели", ANSWER_OPTIONS)
            trust10 = st.radio("Я понимаю почему модель дает определенные результаты", ANSWER_OPTIONS)
            submit_button = st.form_submit_button(label="Отправить анкету")
