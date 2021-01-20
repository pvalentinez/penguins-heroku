import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
**Penguin Prediction App**
""")

st.sidebar.header('User Input Feature')
st.sidebar.markdown("""
[Exmaple CSV input file]
""")

# collect user input featuer into df
upload_file = st.sidebar.file_uploader("Upload input csv file", type=["csv"])
if upload_file is not None:
    input_def = pd.read.csv(upload_file)
else:
    def user_input_feature():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)',32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)',13.1,21.0,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)',172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)',2700.0,6300.0,4207.0)
        data = {'island':island,
                'sex':sex,
                'bill_length_mm':bill_length_mm,
                'bill_depth_mm':bill_depth_mm,
                ' flipper_length_mm': flipper_length_mm,
                'body_mass_g':body_mass_g}
        features = pd.DataFrame(data, index=[0])
    input_df = user_input_feature()

# combine user input with dataset
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns='species')
df = pd.concat([input_df,penguins],axis=0)

# encode get dummie
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]
df = df[:1]

# Display the user input features
st.subheader('User Input features')

if upload_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be upload')
    st.write(df)

# read clf model
load_clf = pickle.load(open('penguins_clf.pkl','rb'))

# apply model
prediction = load_clf.predict(df)
prediction_prob = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction proba')
st.write(prediction_prob)


        