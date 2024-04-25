import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')

# memisahkan data dan label
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# membuat model
scaler = StandardScaler()
scaler.fit(X)
standrized_data = scaler.transform(X)
X = standrized_data
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# judul dalam web
st.title('Data Mining Prediksi Diabates')

# implementasikan kolom
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.text_input('Input Nilai Pregnancies')

with col2:
    Glucose = st.text_input('Input Nilai Glucose')

with col1:
    BloodPressure = st.text_input('Input Nilai Blood Pressure')

with col2:
    SkinThickness = st.text_input('Input Nilai Skin Thickness')

with col1:
    Insulin = st.text_input('Input Nilai Insulin')

with col2:
    BMI = st.text_input('Input Nilai BMI')

with col1:
    DiabetesPedigreeFunction = st.text_input(
        'Input Nilai Diabetes Pedigree Function')

with col2:
    Age = st.text_input('Input Nilai Age')

# kode untuk prediksi
diab_diagnosis = ''

# membuat tombol prediksi
if st.button('Test Status Diabetes'):

    input_data = (Pregnancies, Glucose, BloodPressure,
                  SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)

    input_data_as_numpy_array = np.array(input_data, dtype=float)

    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

    std_data = scaler.transform(input_data_reshape)
    print(std_data)

    prediction = classifier.predict(
        std_data)

    if (prediction[0] == 0):
        diab_diagnosis = 'Anda Tidak Terkena Diabates'
    else:
        diab_diagnosis = 'Anda Terkena Diabates'

    st.success(diab_diagnosis)