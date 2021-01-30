import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.write("""
# Simple Wine Prediction App

This app is to predict wine type.

""")

st.sidebar.header("User Input Parameters")

def user_input_feature():
    alcohol = st.sidebar.slider("Alcohol", 11.0, 14.8, 13.0) 
    malic_acid = st.sidebar.slider("Malic Acid", 0.74, 5.80, 2.34)
    ash = st.sidebar.slider('Ash', 1.36, 3.23, 2.36)
    alcalinity_of_ash = st.sidebar.slider('Alcalinity of Ash', 10.6, 30.0, 19.5)
    magnesium = st.sidebar.slider('Megnesium', 70.0, 162.0, 99.7)
    total_phenols = st.sidebar.slider('Total Phemols', 0.98, 3.88, 2.29)
    flavanoids = st.sidebar.slider('Flavanoids', 0.34, 5.08, 2.03)
    nonflavanoids_phenols = st.sidebar.slider('Nonflavanoid Phenols', 0.13, 0.66, 0.36)
    proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.41, 3.58, 1.59)
    colour_intensity = st.sidebar.slider('Colour_Intensity', 1.3, 13.0, 5.1)
    hue = st.sidebar.slider('Hue', 0.48, 1.71, 0.96)
    diluted_wines = st.sidebar.slider('OD280/OD315 of diluted wines', 1.27, 4.00, 2.61)
    proline = st.sidebar.slider('Proline', 280, 1680, 746)
    data = {'alcohol': alcohol,
            'malic_acis': malic_acid,
            'ash': ash,
            'alcalinity_of_ash': alcalinity_of_ash,
            'magnesium': magnesium,
            'total_phenols': total_phenols,
            'flavanoids': flavanoids,
            'nonflavanoids_phenols': nonflavanoids_phenols,
            'proanthocyanins': proanthocyanins,
            'colour_intensity': colour_intensity,
            'hue': hue,
            'diluted_wines': diluted_wines,
            'proline': proline}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_feature()

st.subheader('User Input Parameters')
st.write(df)

wine = datasets.load_wine()

X = wine.data
Y = wine.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(wine.target_names)

st.subheader('Prediction')
st.write(wine.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_prob)