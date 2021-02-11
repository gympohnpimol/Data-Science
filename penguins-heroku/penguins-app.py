import pandas as pd
import numpy as np 
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App
""")

st.sidebar.header("User Input Features")
st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/gympohnpimol/Data-Science/master/WebApp-for-DataSci/penguin-app/PENGUINS.csv)
    """)

uploaded_file = st.sidebar.file_uploader("uplaod your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
        sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
        bill_length_mm = st.sidebar.slider("Bill length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
        data = {"island": island,
                "bill_length_mm": bill_length_mm,
                "bill_depth_mm": bill_depth_mm,
                "flipper_length_mm": flipper_length_mm,
                "body_mass_g": body_mass_g,
                "sex": sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

penguins_raw = pd.read_csv("/Users/gym/Desktop/Data-Science/WebApp-for-DataSci/penguin-app/PENGUINS.csv")
penguins = penguins_raw.drop(columns=["species"])
df = pd.concat([input_df, penguins], axis=0)

encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df.drop(columns=["sex_MALE"],axis=1)
df = df[:1]
# print(df)
st.subheader("User Input Features")
if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded. Currently input parameters (shown below).")
    st.write(df)

load_clf = pickle.load(open("penguins_clf.pkl", "rb"))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)
# print(penguins_species)
st.subheader("Prediction Probability")
st.write(prediction_proba)

