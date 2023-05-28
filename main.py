import streamlit as st
import pandas as pd
import numpy as np
from Delhi import delhi
from Chennai import chennai
from Kolkata import kolkata
from Mumbai import mumbai
from Hyderabad import hyderabad
from Bangalore import bangalore


st.set_page_config(page_title='House price prediction', page_icon='📈', layout="centered", initial_sidebar_state="auto", menu_items=None)


def main_page():
   
    st.title('Metropolitian areas house price prediction website ✅'    )
    st.sidebar.markdown("Metropolitian areas house price prediction website")
    st.write("Hello, Welcome to the House price prediction website!")
    st.write("People can use this amazing website for predicting and analysing the house price in different area for a particular state. These website includes Metropolitian areas, for instance Mumbai, Hyderabad, Kolkata, Chennai and Delhi.")
    st.write("Some individiual can predict the house on the basis of perimeters like number of bedrooms in the house, square feets , location and facilities like car parking and jogging track.")

page_names_to_funcs = {
    "✅Main Page": main_page,
    "📈Delhi city's house price ": delhi,
    "📈 Chennai city's house price": chennai,
    "📈 Mumbai city's house price": mumbai,
    "📈 Kolkata city's house price": kolkata,
    "📈 Hyderabad city's house price": hyderabad,
    "📈 Bangalore city's house price": bangalore,

    
}
selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()