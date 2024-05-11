import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plost
import pickle
import sklearn
import shap
import matplotlib.pyplot as plt
from streamlit_gsheets import GSheetsConnection
import ssl
import urllib3
import requests
import io

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

conn = st.connection("gsheets", type=GSheetsConnection)
url = "https://docs.google.com/spreadsheets/d/1Izh4WHMK6QCLn_sCOKilM24CH21aDu9VcTzDobKLv6A/edit#gid=0"

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
tab1, tab2, tab3, tab4 = st.tabs(["Персоны", "Дэшборд", "Анкета", "Доп. материалы"])
with tab1:
        st.markdown('### Персоны')
        column1, column2 = st.columns(2)
        with column1:
                st.markdown('##### Сотрудник 1')
                st.image('female.png', width=100)
                st.markdown('**Возраст:** 30 лет')
                st.markdown('**Образование:** высшее (бакалавриат)')
                st.markdown('**Отдел:** финансы')
                st.markdown('**Как давно работает в компании:** 6 лет')
                st.markdown('**Рейтинг за предыдущий год:** 3/5')
                st.markdown('**Кол-во курсов, пройденных в рамках корпоративного обучения за посл. год:** 5')
                st.markdown('**Средняя оценка за пройденные курсы:** 80/100 ')
        with column2:
                st.markdown('##### Сотрудник 2')
                st.image('male.png', width=100)
                st.markdown('**Возраст:** 55 лет')
                st.markdown('**Образование:** высшее (магистратура)')
                st.markdown('**Отдел:** маркетинг')
                st.markdown('**Как давно работает в компании:** 1 год')
                st.markdown('**Рейтинг за предыдущий год:** 4/5')
                st.markdown('**Кол-во курсов, пройденных в рамках корпоративного обучения за посл. год:** 1')
                st.markdown('**Средняя оценка за пройденные курсы:** 60/100 ')
            
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
                st.markdown('##### Как :blue[средний балл за курсы] влияет на предсказание')
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
            trust3 = st.radio("Результаты модели вызывают у меня подозрения", ANSWER_OPTIONS, index=None)
            trust4 = st.radio("Я был(а) бы осторожен(на) по отношению к решениям, принимаемым этой моделью", ANSWER_OPTIONS, index=None)
            trust5 = st.radio("Использование этой модели может привести к вредоносным последствиям", ANSWER_OPTIONS, index=None)
            trust6 = st.radio("Я уверен(а) в результатах модели", ANSWER_OPTIONS, index=None)
            trust7 = st.radio("Результаты модели безопасны для применения", ANSWER_OPTIONS, index=None)
            trust8 = st.radio("На результаты модели можно с уверенностью положиться", ANSWER_OPTIONS, index=None)
            trust9 = st.radio("Я доверяю результатам модели", ANSWER_OPTIONS, index=None)
            trust10 = st.radio("Я понимаю, почему модель дает определенные результаты", ANSWER_OPTIONS, index=None)
            submit_button = st.form_submit_button(label="Отправить анкету")

            if submit_button:
                        answers = pd.DataFrame(
                            [
                                {
                                    "Q1": trust1,
                                    "Q2": trust2,
                                    "Q3": trust3,
                                    "Q4": trust4,
                                    "Q5": trust5,
                                    "Q6": trust6,
                                    "Q7": trust7,
                                    "Q8": trust8,
                                    "Q9": trust9,
                                    "Q10": trust10
                                }
                            ]
                        )
                        # Add the new vendor data to the existing data                      
                        
                        class CustomHttpAdapter(requests.adapters.HTTPAdapter):
                            # "Transport adapter" that allows us to use custom ssl_context.
                        
                            def __init__(self, ssl_context=None, **kwargs):
                                self.ssl_context = ssl_context
                                super().__init__(**kwargs)
                        
                            def init_poolmanager(self, connections, maxsize, block=False):
                                self.poolmanager = urllib3.poolmanager.PoolManager(
                                    num_pools=connections,
                                    maxsize=maxsize,
                                    block=block,
                                    ssl_context=self.ssl_context,
                                )
                        
                        
                        def get_legacy_session():
                            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                            ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
                            session = requests.session()
                            session.mount("https://", CustomHttpAdapter(ctx))
                            return session
                        
                        res = get_legacy_session().get(url)                        
                        st.write(df)
                
                        existing_data = conn.read(spreadsheet=res, worksheet="Answers", usecols=list(range(10)))
                        existing_data = existing_data.dropna(how="all")
                        updated_df = pd.concat([existing_data, answers], ignore_index=True)

                        # Update Google Sheets with the new vendor data
                        conn.update(worksheet="Answers", data=updated_df)
                        st.success("Ответы записаны!")
with tab4:
        st.markdown('### Дополнительные графики')
        # Row A
        c1, c2 = st.columns(2)
        with c1:
                st.markdown('##### Как :blue[возраст] влияет на предсказание')
                shap.dependence_plot("age", shap_values, df_prob, feature_names=df_prob.columns, interaction_index="Вероятность повышения", show = False)
                plt.ylabel("SHAP значения\n для Возраста")
                plt.xlabel("Возраст")
                st.pyplot()
        with c2:
                st.markdown('##### Как :blue[продолжительность работы] влияет на предсказание')
                shap.dependence_plot("length_of_service", shap_values, df_prob, feature_names=df_prob.columns, interaction_index="Вероятность повышения", show = False)
                plt.ylabel("SHAP значения\n для Продолжительности работы")
                plt.xlabel("Продолжительность работы")
                st.pyplot()
        # Row B
        col1, col2 = st.columns(2)
        with col1:
                st.markdown('##### Как :blue[награды] влияют на предсказание')
                shap.dependence_plot("awards_won", shap_values, df_prob, feature_names=df_prob.columns, interaction_index="Вероятность повышения", show = False)
                plt.ylabel("SHAP значения\n для Наград")
                plt.xlabel("Награды")
                st.pyplot()
        with col2:
                st.markdown('##### Как :blue[кол-во пройденных курсов] влияет на предсказание')
                shap.dependence_plot("no_of_trainings", shap_values, df_prob, feature_names=df_prob.columns, interaction_index="Вероятность повышения", show = False)
                plt.ylabel("SHAP значения\n для Кол-ва курсов")
                plt.xlabel("Кол-во курсов")
                st.pyplot()
