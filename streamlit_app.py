import streamlit as st
import pandas as pd
import plost

st.title('Best Dash💖')

st.write('This is gonna be XAI dashboard')

# Row A
st.markdown('### Metrics')

vars = [['education', 'Уровень образования сотрудника'], ['gender', 'Пол сотрудника'], 
        ['no_of_trainings', 'Кол-во тренингов, которые прошел сотрудник за последний год'],
        ['avg_training_score', 'Средняя оценка за пройденные тренинги'] ['age', 'Возраст сотрудника'], 
        ['previous_year_rating', 'Рейтинг сотрудника за прошлый год'],
        ['length_of_service', 'Кол-во лет, которое сотрудник работает в компании'], ['awards_won', 'Кол-во выигранных наград'],
        ['department_', 'Отдел, в котором работает сотрудник']]
vars_df = pd.DataFrame(vars, columns=['Переменная', 'Описание'])

col1, col2 = st.columns(2)
col1.table(vars_df)
col2.metric("Humidity", "86%", "4%")

# Row B
df = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/prom.csv')

c1, c2 = st.columns((7,3))
with c1:
    st.markdown('### Heatmap')
    c1.metric("Temperature", df["age"].mean(), "1.2 °F")
with c2:
    st.markdown('### Donut chart')
    c2.metric("Temperature", df["age"].mean(), "1.2 °F")
