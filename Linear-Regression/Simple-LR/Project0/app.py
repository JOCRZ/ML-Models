import streamlit as st
import pandas as pd
import numpy as np
import pickle

model=pickle.load(open('model.pkl','rb'))
data = pd.read_csv('data/placement.csv')
st.title("LPA Prediction using CGPA")
st.image("data/LPA.jpeg", width=500)


nav = st.sidebar.radio("Navigation",["Aim","Prediction"])      

if nav == 'Aim':
    st.markdown(""" #### Aim of the Project """)
    st.text('Predict Package using the CGPA Score')

    if st.checkbox("Show Table"):
        st.table(data)



if nav == 'Prediction':
    
    st.header('Know Your Package')
    score = st.text_input("CGPA Score")
    choice = st.selectbox(

    'Pick Model or Formula to Predict',

    ('Model','Formula'))
    
    if choice == 'Model':
        def lpa_predict(score):
            val = score
            a = model.predict(val)[0]
            return np.round(a,3)

    if choice == 'Formula':
        def lpa_predict(score):
            m = 0.557951973425072
            b = -0.8961119222429144
            x = score
            y = m * x + b
            return np.round(y[0][0],3)

    if st.button("Predict"):
        value = np.array([[score]], dtype=float)
        output=lpa_predict(value)
        st.success('Your Predicted Package is {}'.format(output))
