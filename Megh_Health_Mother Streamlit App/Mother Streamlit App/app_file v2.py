
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image

st.set_page_config(page_title='Mother App Data Analysis',layout="wide",)

delivery_loc_trend=pd.read_excel("Datasets/Delivery Location Trend.xlsx",index_col=0)
delivery_prof=pd.read_excel("Datasets/Delivery Professional.xlsx",index_col=0)
high_risk_preg_trend=pd.read_excel("Datasets/High Risk Pregnancies trend.xlsx",index_col=0)
scheme_enrol_trend=pd.read_excel("Datasets/Scheme Enrollment Trend.xlsx",index_col=0)
trimester_regist_trend=pd.read_excel("Datasets/Trimester Registrations trend.xlsx",index_col=0)

geo_df=pd.read_excel("Datasets/Delivery Points.xlsx")
edu_level=pd.read_excel("Datasets/Education Level.xlsx")
home_delivery_prof=pd.read_excel("Datasets/Professional conducting Home Deliveries.xlsx")
distance_delivery_point=pd.read_excel("Datasets/Distance from Delivery Points.xlsx")

high_risk_reason=pd.read_excel("Datasets/High Risk Reasons.xlsx")
infant_death_reason=pd.read_excel("Datasets/Reason for Infant Death.xlsx",sheet_name='req')
infant_immunization=pd.read_excel("Datasets/Infant Immunization.xlsx",sheet_name='req')
infant_immunization['Pct']=infant_immunization['Pct'].apply(lambda z:int(np.round(z,0)))

#%%

# Basic Insights

st.markdown('<div style="text-align: center; font-size:30px; font-weight:bold">Basic Insights and Numbers</div>', unsafe_allow_html=True)
st.markdown("\n\n")
st.markdown("\n\n")
col1,col2,col3=st.columns(3)

with col1:    
    st.metric("Number of Mother Registrations in 1st Trimester",1653)
    st.metric("Number of Mother Registrations in 2nd Trimester",3692)
    st.metric("Number of Mother Registrations in 3rd Trimester",570)
    st.metric("Number of High risk Pregnancies",3106)

with col2:
    st.metric('Number of Deliveries',2620)
    st.metric("Number of Home Deliveries",961)
    st.metric("Number of Institutional Deliveries",1192)
    st.metric("Number of New-borns",2651)

    
with col3:
    st.metric("Maternal Deaths",18)
    st.metric("Infant Deaths",46) 
    st.metric("Still Births",43)
    st.metric("Abortions",32)


fishbone=Image.open("Datasets/Fishbone diagram.png")
fishbone=fishbone.resize((700,500))

st.markdown("\n")
st.markdown("\n")
st.markdown('<div style="text-align: center; font-size:18px; font-weight:bold">Understanding the potential causes for Maternal Death</div>', unsafe_allow_html=True)
st.markdown("\n")

e1,e2,e3=st.columns(3)

with e2:
    st.image(fishbone,caption="Cause-Effect for Maternal Death")
    
st.markdown("\n")
st.markdown("\n")

p1,p2=st.columns(2)

with p1:
    st.info("2,620 Deliveries Conducted. Out of them 498 were registered in 1st Trimester.\
            60% of these deliveries received 3 or less ANC visits",icon="ℹ️")
    st.markdown("\n\n\n")
    
    st.info("18 Maternal Deaths are reported. Out of them 7 were High Risk cases. In those 7, 5 were\
            due to teenage or 35+ yrs age. 39% of deaths are of mothers less than 22yrs.",icon="ℹ️")
    st.markdown("\n\n\n")
    
    st.info("47% of pregnancies are high risk cases. 38% (7 out of 18) of maternal deaths are high-risk cases.",icon="ℹ️")
    st.markdown("\n\n\n")
    
    st.info("35% of high risk cases are teenage or 35 yrs+ pregnancies. 60% of women are aged 25 or less at the time of pregnancy.\
            18% of women are aged 19 or less at the time of pregnancy. \
            Almost 9% of women are aged 35 or more at the time of pregnancy. ",icon="ℹ️")
    st.markdown("\n\n\n")

with p2:
    st.info("At time of registration, the HIV test is done for only 25% of the mothers. \
            The VDRL test is done for only 23% of the mothers",icon="ℹ️")
    st.markdown("\n\n\n")
    
    st.info("19% of pregnancies are of order 4 or more (reason for high risk).",icon="ℹ️")
    st.markdown("\n\n\n")
    
    st.info("20% of mothers have weight 42kgs or below at the time of pregnancy.",icon="ℹ️")
    st.markdown("\n\n\n")
    
    st.info("In second last ANC visit, 94% of mothers have Hb less than 10.9 (below recommended level).",icon="ℹ️")
    

with st.expander('Delivery Geo-Locations'):    
    st.map(geo_df,zoom=8)


st.markdown("\n\n")
st.markdown('<div style="text-align: center; font-size:20px; font-weight:bold">Monitoring KPIs</div>', unsafe_allow_html=True)
st.markdown("\n\n")

#%%

with st.expander("The Location of Deliveries"):
    delivery_loc_trend=delivery_loc_trend.fillna(0).applymap(lambda z:int(np.round(z)))
    st.write("Number of deliveries at various locations:")
    st.dataframe(delivery_loc_trend.iloc[:-2,:6],use_container_width=True)
    delivery_loc_trend_display=delivery_loc_trend.iloc[:-2,6:-1]
    delivery_loc_trend_display.reset_index(inplace=True)
    st.markdown("\n")
    bar_ch_sch=px.bar(delivery_loc_trend_display,x='Actual Date of Delivery Qtr',y=['Pct Higher Facility','Pct Home Delivery', 'Pct In Transit', 'Pct PHC/CHC','Pct Sub- Center'],
                        title="FY Quarterly Trend- Pct of Deliveries",text_auto=True)
    st.plotly_chart(bar_ch_sch,use_container_width=True)
    aux_df=delivery_loc_trend.iloc[2:-2,[-1]].reset_index()
    st.markdown("\n")
    bar_ch_sch=px.bar(aux_df,x='Actual Date of Delivery Qtr',y='Pct Govt Institutions',
                        title="FY- Quarterly Trend- Pct of Deliveries at Govt. Institutions",text_auto=True)
    st.plotly_chart(bar_ch_sch,use_container_width=True)
    st.markdown("\n")
    st.info("From 2021-Q2 onwards,the percentage of govt. institutional deliveries have increased gradually. \
            Whereas, the percentage of Home deliveries show a decreasing trend.")
    st.markdown("\n")      
    st.subheader("Distance to Nearest Delivery Point for Mothers")
    c1,c2=st.columns(2)
    with c1:
        fig = px.histogram(distance_delivery_point['Overall- (Distance  in Kms)'].dropna(),nbins=7,text_auto=True) 
        st.plotly_chart(fig,use_container_width=True)
    
    with c2:
        fig = px.histogram(distance_delivery_point['Maternal Death cases- (Distance  in Kms)'].dropna(),nbins=7,text_auto=True) 
        st.plotly_chart(fig,use_container_width=True)
    

with st.expander("The Professional who conducted the Delivery"):
    delivery_prof=delivery_prof.fillna(0).applymap(lambda z:int(np.round(z)))
    st.dataframe(delivery_prof.iloc[:-2,:8],use_container_width=True)
    delivery_prof_display=delivery_prof.iloc[:-2,8:]
    delivery_prof_display.reset_index(inplace=True)
    bar_ch_sch=px.bar(delivery_prof_display,x='Actual Date of Delivery Qtr',y=['Pct ANM', 'Pct Dai', 'Pct Dai - Skilled Birth Attendant','Pct Dai - Unskilled Birth Attendant', 'Pct Medical Officer','Pct Others', 'Pct Staff Nurse'],
                        title="FY Quarterly Trend- Person performing  delivery",text_auto=True)
    st.plotly_chart(bar_ch_sch,use_container_width=True)
    st.subheader("Breakdown of Professional performing Home Deliveries")
    bar_ch_sch=px.bar(home_delivery_prof,x='Who conducted the Delivery',y='Count',
                        title="Distribution",text_auto=True)
    st.plotly_chart(bar_ch_sch,use_container_width=True)
    


with st.expander("Trend for High Risk Pregnancies"):
    high_risk_preg_trend=high_risk_preg_trend.fillna(0).applymap(lambda z:int(np.round(z)))
    st.dataframe(high_risk_preg_trend.iloc[:-2,:3],use_container_width=True)
    high_risk_preg_trend_display=high_risk_preg_trend.iloc[:-2,3:]
    high_risk_preg_trend_display.reset_index(inplace=True)
    bar_ch_sch=px.bar(high_risk_preg_trend_display,x='Date of Registration Qtr',y=['Pct No', 'Pct Yes'],
                        title="FY Quarterly Trend- High Risk Pregnancies",text_auto=True)
    st.plotly_chart(bar_ch_sch,use_container_width=True)
    high_risk_reason['Pct']=high_risk_reason['Pct'].apply(lambda z:int(np.round(z,0)))
    st.markdown("\n")
    st.write("Underestanding the reason for High Risk Preganancies")
    bar_chrt_rsn=px.bar(high_risk_reason,x='High Risk Reasons',y='Pct',
                        title="Top reasons for High Risk Prenancies (Percentage Breakdown)",text_auto=True)
    st.plotly_chart(bar_chrt_rsn,use_container_width=True)

with st.expander("Trend for Mothers who are enrolled in Govt. Schemes"):
    scheme_enrol_trend=scheme_enrol_trend.fillna(0).applymap(lambda z:int(np.round(z)))
    st.dataframe(scheme_enrol_trend.iloc[:-2,:4],use_container_width=True)
    scheme_enrol_trend_display=scheme_enrol_trend.iloc[:-2,4:]
    scheme_enrol_trend_display.reset_index(inplace=True)
    bar_ch_sch=px.bar(scheme_enrol_trend_display,x='Program Registration Qtr',y=['Pct JSY & JSSK', 'Pct PMMVY', 'Pct Neither'],
                        title="FY Quarterly Trend- Mothers registering for Govt. Schemes",text_auto=True)
    st.plotly_chart(bar_ch_sch,use_container_width=True)


with st.expander("Trend for Mothers in which Trimesters were they registered "):
    trimester_regist_trend=trimester_regist_trend.fillna(0).applymap(lambda z:int(np.round(z)))
    st.dataframe(trimester_regist_trend.iloc[:-2,:4],use_container_width=True)
    trimester_regist_trend_display=trimester_regist_trend.iloc[:-2,4:]
    trimester_regist_trend_display.reset_index(inplace=True)
    bar_ch_sch=px.bar(trimester_regist_trend_display,x='Registration FY QTR',y=['Pct 1st Trimester', 'Pct 2nd Trimester', 'Pct 3rd Trimester'],
                        title="FY Quarterly Trend- Mothers registering in the 3 Trimesters",text_auto=True)
    st.plotly_chart(bar_ch_sch,use_container_width=True)

with st.expander("Education Level Among Mothers"):
    a1,a2=st.columns(2)
    with a1:
        bar_chrt=px.bar(edu_level,x='Education Levels',y='Overall',text_auto=True)
        st.plotly_chart(bar_chrt,use_container_width=True)
    with a2:
        edu_level_dropped=edu_level.dropna()
        bar_chrt_death=px.bar(edu_level_dropped,x='Education Levels',y='Maternal Death Cases',text_auto=True)
        st.plotly_chart(bar_chrt_death,use_container_width=True)
    

with st.expander("Reason for Infant Deaths"):
    bar_chrt_death=px.bar(infant_death_reason,x='Reason For Death',y='Count ',text_auto=True)
    st.plotly_chart(bar_chrt_death,use_container_width=True)

with st.expander("Immunization of New-borns"):
    bar_immun=px.bar(infant_immunization,x='Vaccine',y='Pct',text_auto=True)
    st.plotly_chart(bar_immun,use_container_width=True)
    

#%%

st.markdown("\n")
st.caption("Please Note: All analyses have been done on the dataset received from Health Dept. Meghalaya.")
st.header("")
st.markdown('<div style="text-align: center; font-size:16px;">&#169 Copyright 2022 DSAI</div>', unsafe_allow_html=True)
st.markdown("\n")
image = Image.open('Datasets/deepspatial.jpg')
image_1=image.resize((180,30))
st.image(image_1)

















