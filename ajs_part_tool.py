import os
import streamlit as st
import pandas as pd
from streamlit_navigation_bar import st_navbar

# Set the Streamlit page configuration
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="AJS Part Tool",
    page_icon="🔧",
)

# Navigation bar setup
pages = ["Home", "Vendor & Customer Quotes", "List Analysis"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "Untitled design.svg")  # Update logo path

styles = {
    "nav": {
        "background-color": "#0A4786",
        "justify-content": "left",
        "height": "70px",  # Adjust this line to set the height of the navigation bar
        "padding": "2px",  # Adjust padding to control spacing inside the navigation bar
    },
    "img": {
        "padding-left": "40px",
        "padding-right": "14px",
        "width": "70px",  # Add or adjust this line to set the logo width
        "height": "70px",  # Add or adjust this line to set the logo height
    },
    "span": {
        "color": "white",
        "padding": "14px",
        "font-size": "16px",  # Adjust font size
        "font-family": 'Roboto, sans-serif',  # Set font family
        "font-weight": "normal",  # Use "bold" for bold text, or "normal" for regular
    },
    "active": {
        "background-color": "white",
        "color": "#0A4786",
        "font-weight": "bold",  # Set bold for the active text
        "font-size": "16px",  # Adjust font size for active state
        "padding": "14px",
    },
}

options = {
    "show_menu": False,
    "show_sidebar": False,
}

# Render the navigation bar
page = st_navbar(
    pages,
    logo_path=logo_path,
    styles=styles,
    options=options,
)


# Import Excel files
@st.cache_data
def load_data():
    vendor_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/AJS/Data/Spreadsheets/VQ Details Today_TR.xlsx')
    customer_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/AJS/Data/Spreadsheets/CQ_Detail_TODAY_TR.xlsx')
    return vendor_df, customer_df


vendor_df, customer_df = load_data()

# Display content based on the selected page
if page == "Home":
    st.write("Hello Home")
elif page == "Vendor & Customer Quotes":
    st.subheader("Vendor & Customer Quotes")

    # Add search box for part number
    part_number = st.text_input("Search for Part Number (PN)", "")

    # Filter vendor and customer data based on the entered part number
    if part_number:
        filtered_vendor_df = vendor_df[vendor_df['PN'].str.contains(part_number, case=False, na=False)]
        filtered_customer_df = customer_df[customer_df['PN'].str.contains(part_number, case=False, na=False)]

        # Display filtered vendor data
        st.subheader("Vendor Quotes")
        if not filtered_vendor_df.empty:
            st.dataframe(filtered_vendor_df, width=1500)
        else:
            st.write("No vendor quotes found for the given part number.")

        # Display filtered customer data
        st.subheader("Customer Quotes")
        if not filtered_customer_df.empty:
            st.dataframe(filtered_customer_df, width=1500)
        else:
            st.write("No customer quotes found for the given part number.")
    else:
        st.write("Please enter a part number to view vendor and customer quotes.")
elif page == "List Analysis":
    st.write("Hello List Analysis")
