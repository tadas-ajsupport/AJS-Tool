import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_navigation_bar import st_navbar

# Set the Streamlit page configuration
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="AJS Part Tool",
    page_icon="🔧",
)


def plot_quote_histogram(data, part_number, title, color, max_y_value=None):
    # Filter for specific part number
    filtered_df = data[data['PN'] == part_number].copy()  # Ensure a copy is made

    # Ensure datetime format and sort by date
    filtered_df['ENTRY_DATE'] = pd.to_datetime(filtered_df['ENTRY_DATE'])
    filtered_df = filtered_df.sort_values(by='ENTRY_DATE')

    # Create a histogram showing the frequency of quotes over time using Plotly
    fig = px.histogram(filtered_df, x='ENTRY_DATE', nbins=50, title=title, color_discrete_sequence=[color])

    # Customize the layout for better aesthetics
    fig.update_layout(
        title_font_size=16,
        xaxis_title='Entry Date',
        yaxis_title='Number of Quotes',
        xaxis_tickformat='%Y-%m-%d',
        bargap=0.2,
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Make background transparent
        hovermode='x unified',
        yaxis=dict(range=[0, max_y_value] if max_y_value else None)  # Update y-axis range properly
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)


# Before plotting histograms, calculate the maximum y-value to synchronize scaling
@st.cache_data
def get_max_histogram_value(vendor_df, customer_df, part_number):
    vendor_filtered_df = vendor_df[vendor_df['PN'] == part_number]
    customer_filtered_df = customer_df[customer_df['PN'] == part_number]

    vendor_max = vendor_filtered_df['ENTRY_DATE'].value_counts().max() if not vendor_filtered_df.empty else 0
    customer_max = customer_filtered_df['ENTRY_DATE'].value_counts().max() if not customer_filtered_df.empty else 0

    return max(vendor_max, customer_max)


# Navigation bar setup
pages = ["Home", "Vendor & Customer Quotes", "List Analysis"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "/Users/tadas/PycharmProjects/AJS/Design/Untitled design.svg")  # Update logo path

styles = {
    "nav": {
        "background-color": "#0A4786",
        "justify-content": "left",
        "height": "70px",
        "padding": "2px"},
    "img": {
        "padding-left": "40px",
        "padding-right": "14px",
        "width": "70px",
        "height": "70px"},
    "span": {
        "color": "white",
        "padding": "14px",
        "font-size": "16px",
        "font-family": 'Roboto, sans-serif',
        "font-weight": "normal"},
    "active": {
        "background-color": "white",
        "color": "#0A4786",
        "font-weight": "bold",
        "font-size": "16px",
        "padding": "14px"},
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
    vendor_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/Data/VQ Details Today_TR.xlsx')
    customer_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/Data/CQ_Detail_TODAY_TR.xlsx')
    return vendor_df, customer_df


vendor_df, customer_df = load_data()

if page == "Vendor & Customer Quotes":
    # Filter out rows where UNIT_COST is 0 in the vendor data
    vendor_df = vendor_df[vendor_df['UNIT_COST'] != 0]

    # Filter out rows where UNIT_PRICE is 0 in the customer data
    customer_df = customer_df[customer_df['UNIT_PRICE'] != 0]

    st.subheader("Vendor & Customer Quotes")

    # Add search box for part number and condition code filter
    col1, col2 = st.columns(2)

    with col1:
        part_number = st.text_input("Search for Part Number (PN)", "")

    with col2:
        condition_code = st.text_input("Filter by Condition Code (e.g., NE, FN, OH)", "")

    # Apply filters to the data
    if part_number or condition_code:
        # Filter by part number
        filtered_vendor_df = vendor_df[
            vendor_df['PN'].str.contains(part_number, case=False, na=False) if part_number else True
        ]
        filtered_customer_df = customer_df[
            customer_df['PN'].str.contains(part_number, case=False, na=False) if part_number else True
        ]

        # Filter by condition code
        if condition_code:
            filtered_vendor_df = filtered_vendor_df[
                filtered_vendor_df['CONDITION_CODE'].str.contains(condition_code, case=False, na=False)
            ]
            filtered_customer_df = filtered_customer_df[
                filtered_customer_df['CONDITION_CODE'].str.contains(condition_code, case=False, na=False)
            ]

        # Check if any data remains after filtering
        if not filtered_vendor_df.empty or not filtered_customer_df.empty:
            # Calculate ranges and averages
            min_cost = filtered_vendor_df['UNIT_COST'].min() if not filtered_vendor_df.empty else None
            max_cost = filtered_vendor_df['UNIT_COST'].max() if not filtered_vendor_df.empty else None
            avg_cost = filtered_vendor_df['UNIT_COST'].mean() if not filtered_vendor_df.empty else None
            min_price = filtered_customer_df['UNIT_PRICE'].min() if not filtered_customer_df.empty else None
            max_price = filtered_customer_df['UNIT_PRICE'].max() if not filtered_customer_df.empty else None
            avg_price = filtered_customer_df['UNIT_PRICE'].mean() if not filtered_customer_df.empty else None

            # Create a summary dataframe
            summary_df = pd.DataFrame({
                "Min Cost Range": [min_cost],
                "Max Cost Range": [max_cost],
                "Average Cost": [avg_cost],
                "Min Price Range": [min_price],
                "Max Price Range": [max_price],
                "Average Price": [avg_price]
            })

            # Rearrange columns
            summary_df = summary_df[[
                "Min Cost Range", "Max Cost Range", "Average Cost",
                "Min Price Range", "Max Price Range", "Average Price"
            ]]

            # Display the summary dataframe
            st.subheader("Cost and Price Range Summary")
            st.dataframe(summary_df)

            # Create two columns for the box plots
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Vendor Quotes")
                if not filtered_vendor_df.empty:
                    fig_vendor = go.Figure()
                    fig_vendor.add_trace(go.Box(
                        y=filtered_vendor_df['UNIT_COST'],
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.5,
                        name="Unit Cost",
                        marker_color='#8FB8CA'
                    ))
                    fig_vendor.update_layout(title="Unit Cost Distribution")
                    st.plotly_chart(fig_vendor)

            with col2:
                st.subheader("Customer Quotes")
                if not filtered_customer_df.empty:
                    fig_customer = go.Figure()
                    fig_customer.add_trace(go.Box(
                        y=filtered_customer_df['UNIT_PRICE'],
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.5,
                        name="Unit Price",
                        marker_color='#91AC9A'
                    ))
                    fig_customer.update_layout(title="Unit Price Distribution")
                    st.plotly_chart(fig_customer)

            # Check if part number is specified for histograms
            if part_number:
                # Get the maximum y-value for synchronized scaling
                max_y_value = get_max_histogram_value(filtered_vendor_df, filtered_customer_df, part_number)

                # Create two columns for histograms
                col1, col2 = st.columns(2)

                with col1:
                    plot_quote_histogram(
                        filtered_vendor_df,
                        part_number,
                        title="Frequency of Vendor Quotes Over Time",
                        color='#8FB8CA',
                        max_y_value=max_y_value
                    )

                with col2:
                    plot_quote_histogram(
                        filtered_customer_df,
                        part_number,
                        title="Frequency of Customer Quotes Over Time",
                        color='#91AC9A',
                        max_y_value=max_y_value
                    )
