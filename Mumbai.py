import streamlit as st
import pandas as pd
import numpy as np
def mumbai():
    st.title('''Mumbai city's house price prediction''')

    c=pd.read_csv('D:\zeel\projects\metropolitian_house_price_prediction\Mumbai.csv')
    # st.write(c.head())
    c=c[['CarParking','Area','Location','No. of Bedrooms','Price','JoggingTrack']]
    c['Price']=c['Price']/100000

    location_stats = c['Location'].value_counts(ascending=False)
    c1=location_stats[location_stats<=5]
    c.Location = c.Location.apply(lambda x: 'other' if x in c1 else x)

    dummies=pd.get_dummies(c.Location)

    c = pd.concat([c,dummies],axis='columns')

    c=c.drop(['Location'],axis='columns')

    X=c.drop(['Price'],axis='columns')
    y=c['Price']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(X_train,y_train)
    model.score(X_test,y_test)
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)



    columns = {
        'data_columns' : [col.lower() for col in X.columns]
    }
    d=X.columns
    location = st.selectbox(
    'Type or select an area from the dropdown',
    d[4:])
    carparking = st.number_input('carparking',min_value=0,max_value=200,value=0)
    sqft = st.number_input('sqft',min_value=0,max_value=15000,value=0)
    bed = st.number_input('bed',min_value=0,max_value=10,value=0)
    joggingtrack = st.number_input('joggingtrack',min_value=0,max_value=200,value=0)
    def predict_price(location,carparking,sqft,bed,joggingtrack):    
        loc_index = np.where(X.columns==location)[0][0]

        x = np.zeros(len(X.columns))
        x[0] = carparking
        x[1] = sqft
        x[2] = bed
        x[3]=joggingtrack
        if loc_index >= 0:
            x[loc_index] = 1

        return model.predict([x])[0]
    # a=predict_price('other',1, 1310, 3,1)
    # st.write(a)
    if st.button('Predict the price'):
            prediction = predict_price(location,carparking,sqft,bed,joggingtrack)                          
            st.success(prediction)
            st.write("in lakhs")