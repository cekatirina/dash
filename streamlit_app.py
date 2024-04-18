import streamlit as st
import pandas as pd
import plost

st.title('🎈 App Name')

st.write('Hello world!')

# Row A
st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

# Row B
df = pd.read_csv('https://raw.githubusercontent.com/cekatirina/data/master/prom.csv')

c1, c2 = st.columns((7,3))
with c1:
    st.markdown('### Heatmap')
    c1.metric("Temperature", df["age"].mean(), "1.2 °F")
with c2:
    st.markdown('### Donut chart')
    c2.metric("Temperature", df["age"].mean(), "1.2 °F")
