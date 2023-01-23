

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

#%%

st.set_page_config(layout="wide",page_title="Spending Efficiency")
uploaded_file_r = st.file_uploader("Upload the file for Recurring Expenditure in requisite format")
if uploaded_file_r is not None:
    r_dataframe = pd.read_excel(uploaded_file_r)

else:
    st.warning('Please Upload File')


uploaded_file_nr = st.file_uploader("Upload the file for Non-Recurring Expenditure in requisite format")
if uploaded_file_nr is not None:
    nr_dataframe = pd.read_excel(uploaded_file_nr)



model_r = joblib.load("Model_R.sav")
model_nr = joblib.load("Model_NR.sav")

heatmap_data_r=pd.read_excel("R_heatmap.xlsx")
heatmap_data_nr=pd.read_excel("NR_heatmap.xlsx")

feature_imp_r=pd.read_excel("R_FI.xlsx")
feature_imp_r.set_index('Feature',inplace=True)

feature_imp_nr=pd.read_excel("NR_FI.xlsx")
feature_imp_nr.set_index('Feature',inplace=True)


#%%

col1,col2=st.columns(2)

with col1:
    # Displaying the Feature Importances & Heatmap
    st.header("Recurring Expenditure")
    with st.expander("Feature Importance",expanded=True):
        st.subheader("The feature importances are represented below:")
        st.bar_chart(feature_imp_r,use_container_width=True)
    
    
    with st.expander("Correlation Heatmap & Feature Impact Analysis"):
        st.write("A correlation heatmap to show the relationship between features. \
                     More importantly between the Pass Percentage & other features.")
        fig=px.imshow(heatmap_data_r.corr(),color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(" ")
        st.write("Impact of features on Pass Percentage:")
        base_df=pd.DataFrame([heatmap_data_r.iloc[5,:]],columns=heatmap_data_r.columns)
        base_df.drop(columns='Pass_Perce',inplace=True)
        st.write("Lets take a sample school with the following metrics:")
        st.write(base_df)
        feature=st.selectbox("Select a Feature to visualize its impact.",('LP_Reg_per_capita', 'LP_Contr_per_capita',
       'UP_Reg_per_capita', 'UP_Contr_per_capita', 'SHS_Reg_per_capita','SHS_Cont_per_capita', 'HS_Reg_per_capita', 'HS_Cont_per_capita'))
        new_val=st.number_input("Enter the value to change to")
        new_df=base_df.copy()
        new_df[feature]=new_val
        old_pred=model_r.predict(base_df[['LP_Reg_per_capita', 'LP_Contr_per_capita',
       'UP_Reg_per_capita', 'UP_Contr_per_capita', 'SHS_Reg_per_capita',
       'SHS_Cont_per_capita', 'HS_Reg_per_capita', 'HS_Cont_per_capita']])
        new_pred=model_r.predict(new_df[['LP_Reg_per_capita', 'LP_Contr_per_capita',
       'UP_Reg_per_capita', 'UP_Contr_per_capita', 'SHS_Reg_per_capita',
       'SHS_Cont_per_capita', 'HS_Reg_per_capita', 'HS_Cont_per_capita']])
        base_df['Predicted Pass Percentage']=old_pred
        new_df['Predicted Pass Percentage']=new_pred
        show_df=pd.concat([base_df,new_df])
        show_df.index=['Old','New']
        st.write("The Old & New feature values:")
        st.write(show_df[[feature,'Predicted Pass Percentage']])
        st.write("")
        st.write("Visualizing it in graph")
        chart_df=show_df[['Predicted Pass Percentage']]
        line_ch_sch=px.bar(chart_df,  title="Impact of Features")
        st.plotly_chart(line_ch_sch,use_container_width=True)
        
        
        
    
    #%%
    
    # Predictions on Data
    
    X=r_dataframe[['LP_Reg_per_capita', 'LP_Contr_per_capita',
       'UP_Reg_per_capita', 'UP_Contr_per_capita', 'SHS_Reg_per_capita',
       'SHS_Cont_per_capita', 'HS_Reg_per_capita', 'HS_Cont_per_capita']]
    y_preds=model_r.predict(X)
    
    #predictions=pd.DataFrame(y_preds,columns=['Predictions'],index=dataframe['School name'])
    
    r_dataframe['Predicted pass percentage (%)']=y_preds
    st.subheader(" ")
    
    
    #%%
    
    # Option to type the school name to get exact preds:
    
    tab_school,tab_block,tab_district=st.tabs(['School Specific','Block Specific','District Overview'])    
    
    with tab_school:
        
        school=st.text_input("Please type the name of School",value="Sch_11")
        st.write("School "+school+" metrics and Predicted Pass Percentage 2022:")
        st.write(r_dataframe[r_dataframe['School Name']==school].set_index("School Name"))
        
    with tab_block:
        
        block=st.selectbox("Please select a Block",\
                           ('Pynursla', 'Mylliem', 'Mawphlang', 'Mawknyrew', 'Mawsynram'))
        block_df=r_dataframe[r_dataframe['Block']==block]
        st.write("Number of Schools in the block: ",block_df['School Name'].nunique())
        sub_df=block_df[['School Name', 'Block', 'District','Predicted pass percentage (%)']]
        st.write(sub_df.set_index("School Name"))
        st.write(" ")
        st.write("The average predicted pass percentage for the block: ",np.round(sub_df['Predicted pass percentage (%)'].mean(),2),"%")
    
        bar_ch=px.bar(sub_df[['School Name','Predicted pass percentage (%)']].set_index("School Name"),\
                      y='Predicted pass percentage (%)')
    
        st.plotly_chart(bar_ch,use_container_width=True)
        
    
    with tab_district:
        st.write("Number of Schools in the district: ",r_dataframe['School Name'].nunique())
        dis_df=r_dataframe[['School Name', 'Block', 'District','Predicted pass percentage (%)']]
        st.write(dis_df.set_index("School Name"))
        st.write(" ")
        st.write("The average predicted pass percentage for the district: ",np.round(dis_df['Predicted pass percentage (%)'].mean(),2),"%")
    
        bar_ch_2=px.bar(dis_df[['School Name','Predicted pass percentage (%)']].set_index("School Name"),\
                      y='Predicted pass percentage (%)')
    
        st.plotly_chart(bar_ch_2,use_container_width=True)
        
#%%

with col2:
    # Displaying the Feature Importances & Heatmap
    st.header("Non-Recurring Expenditure")
    with st.expander("Feature Importance",expanded=True):
        st.subheader("The feature importances are represented below:")
        st.bar_chart(feature_imp_nr,use_container_width=True)
    
    
    with st.expander("Correlation Heatmap & Feature Impact Analysis"):
        st.write("A correlation heatmap to show the relationship between features. \
                     More importantly between the Pass Percentage & other features.")
        fig=px.imshow(heatmap_data_nr.corr(),color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(" ")
        st.write("Impact of features on Pass Percentage:")
        base_df_nr=pd.DataFrame([heatmap_data_nr.iloc[5,:]],columns=heatmap_data_nr.columns)
        base_df_nr.drop(columns='Pass_Perce',inplace=True)
        st.write("Lets take a sample school with the following metrics:")
        st.write(base_df_nr)
        feature_nr=st.selectbox("Select a Feature to visualize its impact.",('Elect_per_Capita', 'Driw_per_capita', 'Lib_per_capita',
       'Boys_Toile_per_capita', 'Girls_Toil_per_capita', 'Inte_lab_per_capita','ICT_Lab_per_capita', 'School_Bui_per_capita'))
        new_val_nr=st.number_input("Enter the value to change to",key='nr')
        new_df_nr=base_df_nr.copy()
        new_df_nr[feature_nr]=new_val_nr
        old_pred_nr=model_nr.predict(base_df_nr[['Elect_per_Capita', 'Driw_per_capita', 'Lib_per_capita',
       'Boys_Toile_per_capita', 'Girls_Toil_per_capita', 'Inte_lab_per_capita',
       'ICT_Lab_per_capita', 'School_Bui_per_capita']])
        new_pred_nr=model_nr.predict(new_df_nr[['Elect_per_Capita', 'Driw_per_capita', 'Lib_per_capita',
       'Boys_Toile_per_capita', 'Girls_Toil_per_capita', 'Inte_lab_per_capita',
       'ICT_Lab_per_capita', 'School_Bui_per_capita']])
        base_df_nr['Predicted Pass Percentage']=old_pred_nr
        new_df_nr['Predicted Pass Percentage']=new_pred_nr
        show_df_nr=pd.concat([base_df_nr,new_df_nr])
        show_df_nr.index=['Old','New']
        st.write("The Old & New feature values:")
        st.write(show_df_nr[[feature_nr,'Predicted Pass Percentage']])
        st.write("")
        st.write("Visualizing it in graph")
        chart_df_nr=show_df_nr[['Predicted Pass Percentage']]
        line_ch_sch=px.bar(chart_df_nr,  title="Impact of Features")
        st.plotly_chart(line_ch_sch,use_container_width=True)
        
        
        
    
    #%%
    
    # Predictions on Data
    
    X=nr_dataframe[['Elect_per_Capita', 'Driw_per_capita', 'Lib_per_capita',
       'Boys_Toile_per_capita', 'Girls_Toil_per_capita', 'Inte_lab_per_capita',
       'ICT_Lab_per_capita', 'School_Bui_per_capita']]
    y_preds=model_nr.predict(X)
    
    #predictions=pd.DataFrame(y_preds,columns=['Predictions'],index=dataframe['School name'])
    
    nr_dataframe['Predicted pass percentage (%)']=y_preds
    st.subheader(" ")
    
    
    #%%
    
    # Option to type the school name to get exact preds:
    
    tab_school,tab_block,tab_district=st.tabs(['School Specific','Block Specific','District Overview'])    
    
    with tab_school:
        
        school=st.text_input("Please type the name of School",value="Sch_21",key='nr_school')
        st.write("School "+school+" metrics and Predicted Pass Percentage 2022:")
        st.write(nr_dataframe[nr_dataframe['School Name']==school].set_index("School Name"))
        
    with tab_block:
        
        block=st.selectbox("Please select a Block",\
                           ('Pynursla', 'Mylliem', 'Mawphlang', 'Mawknyrew', 'Mawsynram'),key='nr_block')
        block_df=nr_dataframe[nr_dataframe['Block']==block]
        st.write("Number of Schools in the block: ",block_df['School Name'].nunique())
        sub_df=block_df[['School Name', 'Block', 'District','Predicted pass percentage (%)']]
        st.write(sub_df.set_index("School Name"))
        st.write(" ")
        st.write("The average predicted pass percentage for the block: ",np.round(sub_df['Predicted pass percentage (%)'].mean(),2),"%")
    
        bar_ch=px.bar(sub_df[['School Name','Predicted pass percentage (%)']].set_index("School Name"),\
                      y='Predicted pass percentage (%)')
    
        st.plotly_chart(bar_ch,use_container_width=True)
        
    
    with tab_district:
        st.write("Number of Schools in the district: ",nr_dataframe['School Name'].nunique())
        dis_df=nr_dataframe[['School Name', 'Block', 'District','Predicted pass percentage (%)']]
        st.write(dis_df.set_index("School Name"))
        st.write(" ")
        st.write("The average predicted pass percentage for the district: ",np.round(dis_df['Predicted pass percentage (%)'].mean(),2),"%")
    
        bar_ch_2=px.bar(dis_df[['School Name','Predicted pass percentage (%)']].set_index("School Name"),\
                      y='Predicted pass percentage (%)')
    
        st.plotly_chart(bar_ch_2,use_container_width=True)



#%%

# Displaying Disclaimer & Header

st.write("")
st.caption("Please Note: For demo purposes we have considered a subset of features only.\
         The accuracy of prediction completely relies on the quality and ccompleteness of the dataset.\
        All datasets used are for showcase purpose only.")
st.header(" ")
from PIL import Image
image = Image.open('deepspatial.jpg')
image_1=image.resize((180,30))
st.image(image_1)



