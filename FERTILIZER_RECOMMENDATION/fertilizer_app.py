import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
from utils.fertilizer import fertilizer_dic
#output1= st.read('../utils/fertilizer/fertilizer_dic')
#%%
st.title("DSAI Digital Agriculture Platform")
html_temp = """
    <div style="background-color:orange;padding:6px">
    <h2 style="color:white;text-align:center;">Fertilizer Recommendation </h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
#st.set_page_config(page_title="Fertilizer Recommendation")

with st.form(key='myform'):
    standard_to = StandardScaler()
    
    df = pd.read_csv('fertilizer.csv',index_col=0)
    
    
    N = st.number_input("Nitrogen",step=1)
    P = st.number_input("Phosporus",step=1)
    K = st.number_input("Pottasium",step=1)
    crop_name = st.selectbox("Crop Name",tuple(df.index))
    nr = df.loc[crop_name,'N']
    pr = df.loc[crop_name,'P']
    kr = df.loc[crop_name,'K']
    
    
    n = nr - N
    p = pr - P
    k = kr - K
    
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"
        
    st.header(" ")
    submit_button=st.form_submit_button(label='Click for Fertilizer Recommendation!')    

    if submit_button:
        st.header(" ")
        st.header(" ")
        st.markdown(fertilizer_dic[key], unsafe_allow_html=True)
    

#%%











