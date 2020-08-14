import streamlit as st
import requests
from PIL import Image
import io
import joblib
import pandas as pd
from RLRMC_fapi import predict

st.markdown('<style>  body {background-color: black; color: black}</style>', unsafe_allow_html=True)
choices = [' ','EDA & Visualizations','Customer Segmentation','Customer Lifetime Value','Next Purchase Day','Recommendation System']
selection = st.sidebar.selectbox("Select Your Choice",choices)
#http://127.0.0.1:8000/predict/?user_id=17375&item_id=22325
def get_data(user_id,pro_id):
    server_url = url + endpoint + '?user_id='+ str(user_id) + '&item_id=' + str(pro_id)
    r= requests.get(server_url)
    return r.json()

st.markdown('<style>body{background-color: cornsilk;}</style>',unsafe_allow_html=True)
html_temp = """
<div style="background-color:black;padding:10px">
<h1 style="color:red;text-align:center;">E-Commerce Analysis </h1>
</div> """
st.markdown(html_temp,unsafe_allow_html=True)

if selection == ' ': 
    st.write('') 
    image = Image.open('Argos.png')
    st.image(image, caption='Argos',use_column_width=True)
    st.write('Argos Limited, trading as Argos, is a catalog retailer operating in the United Kingdom and Ireland, and a subsidiary of Sainsbury. The company trades both through physical shops and online, with over 883 retail shops, 29 million yearly shop customers, and nearly a billion online visitors per annum, making it one of the largest high street retailers in the United Kingdom.')


if selection == 'EDA & Visualizations':
    html_temp = """
    <h2 style="color:black;text-align:center;"> </h1>
    </div> """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header('Dashboard Using Salesforce Einstein Analytics')
    days = Image.open('dashboard.png')
    st.image(days,use_column_width=True)
    st.header('Visualization based on Month, Day and Hour')
    month = Image.open('Months.jpg')
    st.image(month,use_column_width=True)
    day = Image.open('Days.jpg')
    st.image(day,use_column_width=True)
    hour = Image.open('Hours.jpg')
    st.image(hour,use_column_width=True)
    st.header('Visualization based spendings and number of orders (Country-wise)')
    order1 = Image.open('countries_orders.jpg')
    st.image(order1,use_column_width=True)
    order2 = Image.open('countries_orders_woUK.jpg')
    st.image(order2,use_column_width=True)
    spent1 = Image.open('spendings_Countries.jpg')
    st.image(spent1,use_column_width=True)
    spent2 = Image.open('Spending_Countries_woUK.jpg')
    st.image(spent2,use_column_width=True)
    st.header('Insights:')
    st.write('1.The customer with the highest number of orders(~181K) comes from the United Kingdom (UK)')
    st.write('2.The customer with the highest money spent on purchases(~40K) comes from Netherlands')
    st.write('3.November 2011 has the highest sales')
    st.write('4.The number of orders received by the company tends to increases from Monday to Thursday and decrese afterward. There are no transactions on Saturday')
    st.write('5.The company receives the highest number of orders at 12:00pm')
    st.write('6.The company receives the highest number of orders from customers in the UK(since it is a UK-based company)')
    st.write('Other top companies other than UK are Germnay, France, Ireland and Spain')
    st.write('7.The customers in the UK spend the most on their purchases.Other top countries that spend the most money on purchases are Netherlands, Ireland, Germany and France ')
    st.write('8.The product "WORLDWAR 2 GLIDERS ASSTD DESIGNS" was sold the most number of times (49709 times) and the product "REGENCY CAKESTAND 3 TIER" generated the highest revenue(21K)')


if selection == 'Recommendation System':
    st.header('Recomendtaion System using Riemannian Low-rank Matrix Completion(RLRMC) algorithm')
    title = st.number_input('Enter Customer ID',min_value = 0,max_value=100000,value = 0,step =1)
    pro = st.number_input('Enter Product ID',min_value = 0,max_value=1000000,value = 0,step =1)
    url = 'http://127.0.0.1:8000/'
    endpoint = 'predict/'
    if st.button('Get Predictions'):
        segments = get_data(title,pro)
        st.write(segments)
        data = pd.read_csv('RLRMC_final.csv')  
        df1 =  data['userID']==title
        df2 =data[df1]
        if df2.empty:
            st.write('we have no information on this user!')
        else:
            df2 = df2.sort_values(by = 'prediction', ascending = False).head(5)
            df2 = df2[['itemID','Description','prediction']].reset_index(drop = True)
            st.write(df2)

if selection == 'Customer Segmentation':
    st.header('Customer Segmentation Using RFM Analysis')
    stats = pd.read_csv('RFM_Stats.csv')
    if st.button("Get RFM stats"):
        st.write(stats)
        cs1 = Image.open('RFM.png')
        st.write('Tree Map showing Customer Segmentation:')
        st.image(cs1, caption='Customer Segmentation',use_column_width=True)
    cs= pd.read_csv('RFM_Segmentation.csv')
    cs['CustomerID'] = cs['CustomerID'].astype(int)
    customerID = st.number_input('Enter Customer ID',min_value = 0,max_value=100000,value = 0,step =1)
    if st.button("Get Customer Segmentation"):
        selected_customer = cs.loc[cs['CustomerID']==customerID]
        st.write(selected_customer)

if selection == 'Customer Lifetime Value':
    
    st.header('Customer Lifetime Value')
    cltv = Image.open('CLTV.jpg')
    st.image(cltv, caption='Customer Lifetime Value',use_column_width=True)
    st.write('Customer Lifetime Value Segmentation (Low-value customers, Mid-value customers and High-value customers) is based on total RFM scores of the customers')
    cs= pd.read_csv('CLV.csv')
    cs['CustomerID'] = cs['CustomerID'].astype(int)
    customerID = st.number_input('Enter Customer ID',min_value = 0,max_value=100000,value = 0,step =1)
    if st.button("Get CLV"):
        selected_customer = cs.loc[cs['CustomerID']==customerID]
        st.write(selected_customer)

if selection == 'Next Purchase Day':
    st.header('Predicting Next Purchase Day')
    st.write('We have categorized the Customers into three categories:')
    st.write('1. 0–20: Customers that will purchase in 0–20 days — Class name: 2')
    st.write('2. 21–49: Customers that will purchase in 21–49 days — Class name: 1')
    st.write('3. ≥ 50: Customers that will purchase in more than 50 days — Class name: 0')
    cs= pd.read_csv('NextPurchase.csv')
    cs['CustomerID'] = cs['CustomerID'].astype(int)
    customerID = st.number_input('Enter Customer ID',min_value = 0,max_value=100000,value = 0,step =1)
    if st.button("Get Next Purchase Day"):
        selected_customer = cs.loc[cs['CustomerID']==customerID]
        if selected_customer.empty:
            st.write('We have no information on this user!')
            
        else:
            
            st.write(selected_customer)
        #st.write(selected_customer)