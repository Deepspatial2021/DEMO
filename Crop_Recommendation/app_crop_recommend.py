# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:25:23 2023

@author: Bushra
"""


#from flask import Flask, render_template, request, Markup
import pickle
import streamlit as st
import sklearn
from sklearn.preprocessing import StandardScaler


crop_recommendation_model = pickle.load(open('rf_crop_reccomend.pkl', 'rb'))

def welcome():
    return "Welcome All"


    
   
def crop_recommend(N, P, K, temperature, humidity, ph, rainfall):
    
    """Let's Predict the Crop Yield 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: N
        in: query
        type: number
        required: true
      - name: P
        in: query
        type: number
        required: true
      - name: K
        in: query
        type: number
        required: true
      - name: temperature
        in: query
        type: number
        required: true
      - name: humidity
        in: query
        type: number
        required: true
      - name: ph
        in: query
        type: number
        required: true
      - name: rainfall
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=crop_recommendation_model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    print(prediction)
    return prediction 

def main():
    st.title("DSAI Digital Agriculture Platform")
    html_temp = """
    <div style="background-color:orange;padding:10px">
    <h2 style="color:white;text-align:center;">Crop Recommendation</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    N = st.text_input("nitrogen")
    P = st.text_input("phosphorous")
    K = st.text_input("pottasium")
    temperature = st.text_input("temperature")
    humidity = st.text_input("humidity")
    ph = st.text_input("ph")
    rainfall = st.text_input("rainfall")
    result=""

    if st.button("Predict"):
        result=crop_recommend(N, P, K, temperature, humidity, ph, rainfall)
    st.success('You can grow [ {} ] on your farm'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")
           
if __name__=='__main__':
    main()    
