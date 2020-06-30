import pickle
import numpy as np
import streamlit as st

pickle1 = open('model.pickle','rb')
classifier = pickle.load(pickle1)

pickle2 = open('X.pickle','rb')
X = pickle.load(pickle2)

pickle3 = open('accuracy.pickle','rb')
accuracy = pickle.load(pickle3)

html = """ <div style='background-color:#ccffeb;padding:10px'>
<h2 style="color:#ff0000;text-align:center;"><b><i>Diabetes Prediction System
</h2></div>"""

st.text('@author: Koshal')
st.markdown(html, unsafe_allow_html=True)
st.header('')

st.subheader('Number of Pregnancies')
a = st.number_input('',value=1, max_value=12, min_value = 0)

st.subheader('Glucose Level')
b = st.number_input(' ',value=60, max_value=350, min_value = 12)

st.subheader('Blood Pressure level')
c = st.number_input('  ',value=50, max_value=200, min_value = 40)

st.subheader('BMI')
d = st.number_input('   ',value=20.0, max_value=55.0, min_value = 15.0)

st.subheader('Age')
e = st.number_input('    ',value=20, max_value=120, min_value = 1)


def predict(preg,glu,bp,bmi,age):
    
    x = np.zeros(len(X.columns))
    x[0] = preg
    x[1] = glu
    x[2] = bp
    x[3] = bmi
    x[4] = age
    
    return classifier.predict([x])[0]

st.header('')

html2 = """ <div style='background-color:white;padding:10px'>
<h2 style="color:red;text-align:center;"><b><i>Based on input data it seems You are <u>SUFFERING</u> from diabetes...
</h2></div>"""

html3 = """ <div style='background-color:white;padding:10px'>
<h2 style="color:green;text-align:center;"><b><i>Based on input data it seems You are <u>SAFE!!
</h2></div>"""

but = st.button('Predict')
if but:
    if predict(abs(a),abs(b),abs(c),abs(d),abs(e)) == 1:
        st.markdown(html2, unsafe_allow_html=True)
        
    else:
        st.markdown(html3, unsafe_allow_html=True)

st.sidebar.header('*PREDICTOR INFO*')
st.sidebar.subheader('')
cb = st.sidebar.checkbox('Technique Used')
if cb:
    st.sidebar.markdown("""<h5 style='color:red;text-align:left'>ENSEMBLE LEARNING</h5>""", unsafe_allow_html=True)

st.sidebar.subheader('')
cb2 = st.sidebar.checkbox('Algorithms Used')
if cb2:
    st.sidebar.markdown("""<h5 style='color:red;text-align:left'>LOGISTIC REGRESSION</h5>
                        <h5 style='color:red;text-align:left'>RANDOM FOREST CLASSIFIER</h5>""", 
                        unsafe_allow_html=True)

st.sidebar.subheader('')
cb3 = st.sidebar.checkbox('Model Accuracy')
if cb3:
    st.sidebar.subheader('{:.3f}'.format(accuracy))


