import streamlit as st
import pandas as pd
import joblib
st.title("Batch Prediction App")
uploaded_file=st.file_uploader("churn-bigml-20.csv",type=["csv"])
model=joblib.load("Churn_prediction.joblib")
if uploaded_file:
    df=pd.read_csv(uploaded_file)
    st.write("Data preview",df.head(50))

    if st.button("predict"):
        predictions=model.predict(df)
        st.write("Predictions:",predictions)
        # df["Predictions"]=predictions
        # st.download_button("Download Result",df.to_csv(index=False),"Result.csv")
