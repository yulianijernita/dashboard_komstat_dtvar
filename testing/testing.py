import streamlit as st

st.title("Testing Page")
with st.form("key=my_form"):
    nama = st.text_input("Enter your name")
    jenis_kelamin = st.radio("jenis kelamin", options=["Male", "Female"])
    checkbox_val = st.checkbox("Form checkbox")
    submit= st.form_submit_button(label="Submit")