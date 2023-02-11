#pip install prophet
#pip install openpyxl
#pip install streamlit
#pip install plotly
    
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly,plot_components
import base64

st.title('Gold Price Forecasting')
st.image('https://g.foolcdn.com/editorial/images/592733/green-arrow-over-gold-bars.jpg',width=600)

st.write('''This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast
''')

st.sidebar.image('https://st4.depositphotos.com/1000423/21642/i/600/depositphotos_216422978-stock-photo-improving-sales-figures.jpg')

st.sidebar.write('Import Data')

df= st.sidebar.file_uploader('Upload here',type='xlsx')
if df is not None:
    data = pd.read_excel(df)
    data['date'] = pd.to_datetime(data['date'],errors='coerce') 
    st.text('Actual data:')
    st.write(data)
    max_date = data['date'].max()
    #st.write(max_date)

st.sidebar.write("Select Forecast Period")

periods_input = st.sidebar.number_input('How many periods would you like to forecast into the future?',
min_value = 1, max_value = 1000)

if df is not None:
    data.columns=['ds','y']
    m = Prophet()
    m.fit(data)



st.subheader('Visualize Forecast Data')

if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']]

    forecast_price =  fcst[fcst['ds'] > max_date] 
    st.text('Forecated price:')
    st.write('The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.')   
    st.write(forecast_price)

    line_chart= plot_plotly(m,forecast)
    st.text('Line chart:')
    st.write("The next visual shows the actual (black dots) and predicted (blue line) values over time.")
    st.write(line_chart)

    components= m.plot_components(forecast)
    st.text('Seasonal components:')
    st.write("The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.")
    st.write(components)

    

st.subheader('Download the Forecast Data')
st.write('The below link allows you to download the newly created forecast data to your computer for further analysis and use.')

if df is not None:
    csv_exp = forecast_price.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)

