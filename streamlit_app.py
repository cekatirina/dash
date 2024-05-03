import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plost
import pickle
import sklearn
import shap
import matplotlib.pyplot as plt

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

df = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/X_test.csv')
y = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/y_test.csv')
df.iloc[[1432],[2]] = 5

modelGB = pickle.load(open('modelGB.pkl', 'rb'))
prediction = modelGB.predict(df)
prediction_proba = modelGB.predict_proba(df)

df_prob = df
df_prob["Вероятность повышения"] = prediction_proba[:,1]

names = ['Образование', 'Отдел: HR', 'Кол-во курсов', 'Возраст', 
         'Рейтинг', 'Продолжительность работы', 'Отдел: Юридический', 'Отдел: Финансы', 'Пол',
         'Награды', 'Отдел: Операционный', 'Отдел: Закупки', 'Отдел: Исследования и разработки',
         'Отдел: Продажи & Маркетинг',  'Отдел: Технологии', 'Реферальная программа', 'Обычный поиск',
         'Ср. балл за курсы', 'prom']

explainer = shap.Explainer(modelGB)
shap_values = explainer.shap_values(df_prob)

example1 = df_prob.iloc[1432]
example1 = example1.to_numpy()

example2 = df_prob.iloc[725]
example2 = example2.to_numpy()

tab1, tab2, tab3 = st.tabs(["Персоны", "Дэшборд", "Анкета"])

with tab1:
        st.markdown('### Персоны')
        column1, column2 = st.columns(2)
        with column1:
                st.markdown('##### Сотрудник 1')
                st.image('female.png', width=400)
        with column2:
                st.markdown('##### Сотрудник 2')
                st.image('male.png')
            
with tab2:
        st.markdown('### Общая статистика по модели')
        
        # Row A
        vars = [['Образование', 'Уровень образования сотрудника (1 - основное общее, 2 - бакалавриат, 3 - магистратура и выше)'], 
                ['Пол', 'Пол сотрудника (0 - мужчина, 1 - женщина)'], 
                ['Кол-во курсов', 'Кол-во курсов, пройденных в рамках корпоративного обучения за посл. год'],
                ['Ср. балл за курсы', 'Средняя оценка за пройденные курсы (макс. 100 баллов)'], ['Возраст', 'Возраст сотрудника'], 
                ['Рейтинг', 'Рейтинг сотрудника за прошлый год (макс. 5)'],
                ['Продолжительность работы', 'Кол-во лет, которое сотрудник работает в компании'], ['Награды', 'Кол-во выигранных наград'],
                ['Отдел:', 'Отдел, в котором работает сотрудник'], ['Реферальная программа', 'Человек устроился в компанию через реферальную программу'], 
                ['Обычный поиск', 'Человек устроился в компанию через обычный поиск сотрудников']]
        vars_df = pd.DataFrame(vars, columns=['Переменная', 'Описание'])
        col1, col2 = st.columns(2)
        with col1:
                st.markdown('##### Описание переменных', help = 'Данные содержат информацию о 40 тыс. сотрудниках и о том, повышались ли они в должности за последний год.')
                st.table(vars_df)
        with col2:
                st.markdown('##### Важность переменных', help = 'Чем дальше SHAP значение от нуля, тем больше переменная влияет на итоговое предсказание')
                st.markdown('Какие переменные повлияли на предсказания модели больше всего?')
                shap.summary_plot(shap_values, df_prob, plot_type='bar', feature_names = names, max_display=18, show = False)
                plt.xlabel("Среднее SHAP значение")
                st.pyplot()
        
        # Row B
        c1, c2 = st.columns(2)
        with c1:
                st.markdown('##### Как :blue[средняя оценка за обучение] влияет на предсказание')
                shap.dependence_plot("avg_training_score", shap_values, df_prob, feature_names=df_prob.columns, interaction_index="Вероятность повышения", show = False)
                plt.ylabel("SHAP значения\n для Ср. балла за курсы")
                plt.xlabel("Ср. балл за курсы")
                st.pyplot()
        with c2:
                st.markdown('##### Как :blue[рейтинг] за предыдущий год влияет на предсказание')
                shap.dependence_plot("previous_year_rating", shap_values, df_prob, feature_names=df_prob.columns, interaction_index="Вероятность повышения", show = False)
                plt.ylabel("SHAP значения\n для Рейтинга")
                plt.xlabel("Рейтинг")
                st.pyplot()

        # Row C
        st.markdown('### Информация по индивидуальным предсказаниям')
        c1, c2 = st.columns(2)
        with c1:
                st.markdown('##### Пример 1 (вероятность повышения: 79%)')
                shap.waterfall_plot(shap.Explanation(values=shap_values[1432],
                            base_values=explainer.expected_value[0],
                            data=example1,
                            feature_names=names), show = False)
                plt.xlabel("SHAP значение")
                st.pyplot()
        with c2:
                st.markdown('##### Пример 2 (вероятность повышения: 58%)')
                shap.waterfall_plot(shap.Explanation(values=shap_values[725],
                            base_values=explainer.expected_value[0],
                            data=example2,
                            feature_names=names), show = False)
                plt.xlabel("SHAP значение")
                st.pyplot()

with tab3:
        st.markdown('### Ваши впечатления от дэшборда')
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
            st.markdown('##### Доверие модели')
            st.markdown('Оцените каждое утверждение по шкале от 1 до 7 (1 - полностью НЕ согласен, 7 - полностью согласен)')
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)    
            trust1 = st.radio("Модель обманчива", ANSWER_OPTIONS, index=None) 
            trust2 = st.radio("Модель принимает решения нечестно", ANSWER_OPTIONS, index=None)
            trust3 = st.radio("Намерения модели вызывают у меня подозрения", ANSWER_OPTIONS, index=None)
            trust4 = st.radio("Я был бы осторожен по отношению к решениям, принимаемым этой моделью", ANSWER_OPTIONS, index=None)
            trust5 = st.radio("Результаты модели могут привести к вредоносным последствиям", ANSWER_OPTIONS, index=None)
            trust6 = st.radio("Я уверен в результатах модели", ANSWER_OPTIONS, index=None)
            trust7 = st.radio("Результаты модели безопасны для применения", ANSWER_OPTIONS, index=None)
            trust8 = st.radio("На результаты модели можно с уверенностью положиться", ANSWER_OPTIONS, index=None)
            trust9 = st.radio("Я доверяю результатам модели", ANSWER_OPTIONS, index=None)
            trust10 = st.radio("Я понимаю почему модель дает определенные результаты", ANSWER_OPTIONS, index=None)
            submit_button = st.form_submit_button(label="Отправить анкету")
