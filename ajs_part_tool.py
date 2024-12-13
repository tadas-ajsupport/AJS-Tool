# --- REQUIRED ADDONS --- #
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_navigation_bar import st_navbar

# --- PAGE CONFIGURATION --- #
st.set_page_config(page_title="AJS Part Tool", page_icon="🔧", layout="wide")


# --- FUNCTIONS --- #
@st.cache_data
def load_data():
    vendor_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/Data/VQ Details Today_TR.xlsx')
    customer_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/Data/CQ_Detail_TODAY_TR.xlsx')
    quote_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/Data/quote_df.xlsx')
    pn_master_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/Data/pn_master.xlsx')
    stock_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/Data/stock_df_original.xlsx')
    sales_df = pd.read_excel('/Users/tadas/PycharmProjects/AJS/Data/sales_df.xlsx')

    return vendor_df, customer_df, quote_df, pn_master_df, stock_df, sales_df


vendor_df, customer_df, quote_df, pn_master_df, stock_df, sales_df = load_data()


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


def convert_to_date(df, column):
    df[column] = pd.to_datetime(df[column]).dt.date


convert_to_date(vendor_df, 'ENTRY_DATE')
convert_to_date(customer_df, 'ENTRY_DATE')
convert_to_date(sales_df, 'ENTRY_DATE')
quote_df['TIMESTAMP'] = pd.to_datetime(quote_df['TIMESTAMP'])
quote_df['TIMESTAMP'] = quote_df['TIMESTAMP'].dt.date

# --- NAVIGATION BAR --- #
pages = ["Home", "Vendor & Customer Quotes", "List Analysis"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "/Users/tadas/PycharmProjects/AJS/Design/Untitled design.svg")  # Update logo path

styles = {
    "nav": {
        "background-color": "#0A4786",
        "justify-content": "left",
        "height": "80px",
        "padding": "2px"},
    "img": {
        "padding-left": "40px",
        "padding-right": "14px",
        "width": "100px",
        "height": "100px"},
    "span": {
        "color": "white",
        "padding": "14px",
        "font-size": "16px",
        "font-family": 'sans-serif',
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

# --- HOME PAGE --- #
if page == "Home":
    # Ensure 'PN' and other columns are of type string for processing
    quote_df['PN'] = quote_df['PN'].astype(str)
    quote_df['QTY'] = quote_df['QTY'].astype(str)

    # Function to split PN, Description, and QTY columns
    def split_row(row):
        # Split the columns into lists
        pns = row['PN'].split(', ')
        descriptions = row['Description'].split(', ') if isinstance(row['Description'], str) else [''] * len(pns)
        qtys = row['QTY'].split(', ') if isinstance(row['QTY'], str) else [row['QTY']] * len(pns)

        # Align descriptions and qtys with the length of PNs
        descriptions = descriptions[:len(pns)] + [''] * (len(pns) - len(descriptions))
        qtys = qtys[:len(pns)] + [''] * (len(pns) - len(qtys))

        return pd.DataFrame({
            'PN': pns,
            'Description': descriptions,
            'QTY': qtys
        })

    # Process rows with multiple part numbers
    split_rows = []
    qt_counter = 1  # Initialize QT# counter
    for _, row in quote_df.iterrows():
        if ', ' in row['PN']:
            split_df = split_row(row).assign(
                **{col: row[col] for col in quote_df.columns if col not in ['PN', 'Description', 'QTY']}
            )
            split_df['QT#'] = qt_counter  # Assign the same QT# to all split rows
            split_rows.append(split_df)
            qt_counter += 1  # Increment QT# for the next quote
        else:
            row_df = pd.DataFrame([row])
            row_df['QT#'] = qt_counter  # Assign QT# to the single row
            split_rows.append(row_df)
            qt_counter += 1  # Increment QT# for the next quote

    # Concatenate all rows back together
    processed_df = pd.concat(split_rows, ignore_index=True)

    # Move 'QT#' to the first column
    cols = ['QT#'] + [col for col in processed_df.columns if col != 'QT#']
    processed_df = processed_df[cols]

    # Add "Stock" column
    processed_df['Stock'] = processed_df['PN'].apply(lambda x: '' if x in stock_df['PN'].values else ' ')

    # Add "PN Master" column
    processed_df['PN Master'] = processed_df['PN'].apply(lambda x: '' if x in pn_master_df['PN'].values else ' ')

    # Add "Last Sale Date" column
    def get_last_sale_date(part_number):
        # Filter sales_df for rows matching the part number
        matching_sales = sales_df[sales_df['PN'] == part_number]
        if not matching_sales.empty:
            return matching_sales['ENTRY_DATE'].max()  # Return the most recent sale date
        return None  # No sales found

    processed_df['Last Sale Date'] = processed_df['PN'].apply(get_last_sale_date)

    # Add "Last Unit Cost" and "Last Unit Price" columns
    def get_last_unit_cost(part_number):
        # Filter sales_df for rows matching the part number
        matching_sales = sales_df[sales_df['PN'] == part_number]
        if not matching_sales.empty:
            return round(matching_sales.loc[matching_sales['ENTRY_DATE'].idxmax(), 'UNIT_COST'], 2)  # Get UNIT_COST for the most recent sale and round
        return ''  # No sales found

    def get_last_unit_price(part_number):
        # Filter sales_df for rows matching the part number
        matching_sales = sales_df[sales_df['PN'] == part_number]
        if not matching_sales.empty:
            return round(matching_sales.loc[matching_sales['ENTRY_DATE'].idxmax(), 'UNIT_PRICE'], 2)  # Get UNIT_PRICE for the most recent sale and round
        return ''  # No sales found

    processed_df['Last Unit Cost'] = processed_df['PN'].apply(get_last_unit_cost)
    processed_df['Last Unit Price'] = processed_df['PN'].apply(get_last_unit_price)

    # Add "Pst 6M VQTs" and "Pst 6M CQTs" columns
    def get_past_6m_vendor_quotes(part_number):
        six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
        matching_vendor_quotes = vendor_df[
            (vendor_df['PN'] == part_number) &
            (pd.to_datetime(vendor_df['ENTRY_DATE']) >= six_months_ago)
            ]
        return len(matching_vendor_quotes)


    def get_past_6m_customer_quotes(part_number):
        six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
        matching_customer_quotes = customer_df[
            (customer_df['PN'] == part_number) &
            (pd.to_datetime(customer_df['ENTRY_DATE']) >= six_months_ago)
            ]
        return len(matching_customer_quotes)

    processed_df['Pst 6M VQTs'] = processed_df['PN'].apply(get_past_6m_vendor_quotes)
    processed_df['Pst 6M CQTs'] = processed_df['PN'].apply(get_past_6m_customer_quotes)

    # Replace None or NaN with an empty string in relevant columns
    processed_df[['Last Unit Cost', 'Last Unit Price']] = processed_df[['Last Unit Cost', 'Last Unit Price']].fillna('')
    non_numeric_cols = ['Stock', 'PN Master', 'Last Sale Date', 'Description', 'QTY']
    processed_df[non_numeric_cols] = processed_df[non_numeric_cols].fillna('')

    # Ensure numeric columns remain numeric
    processed_df[['Last Unit Cost', 'Last Unit Price']] = processed_df[['Last Unit Cost', 'Last Unit Price']].apply(pd.to_numeric, errors='coerce')

    # Function to style the cells based on "Stock" and "PN Master"
    def color_cells(value):
        if value == '':
            return 'background-color: #7ABD7E; color: white;'
        elif value == ' ':
            return 'background-color: #FF6961; color: white;'
        return ''

    # Function to style the "Last Sale Date" column
    def style_last_sale_date(value):
        if value == '':  # Check if the value is empty (no sale found)
            return 'background-color: grey; color: white;'
        return ''  # No styling for cells with sale dates

    # Function to style "Last Unit Cost" and "Last Unit Price" columns
    def style_last_unit_values(value):
        if pd.isna(value) or value == '':  # Check if the value is NaN or empty
            return 'background-color: grey; color: white;'
        return ''

    # Function to style "Pst 6M VQTs" and "Pst 6M CQTs" columns based on proximity to 150
    def style_quote_counts(value):
        if value > 0:
            base_intensity = 100  # Set the minimum green intensity
            green_intensity = min(255, base_intensity + int(
                (value / 150) * (255 - base_intensity)))  # Scale between base_intensity and 255
            return f'background-color: rgb(0, {green_intensity}, 0); color: white;'
        return ''

    # Apply styling to the dataframe
    styled_df = processed_df.style.applymap(color_cells, subset=['Stock', 'PN Master'])
    styled_df = styled_df.format({
        'Last Unit Cost': lambda x: f'{x:.2f}' if not pd.isna(x) and x != '' else '',  # Conditional formatting
        'Last Unit Price': lambda x: f'{x:.2f}' if not pd.isna(x) and x != '' else ''  # Conditional formatting
     }).applymap(color_cells, subset=['Stock', 'PN Master']) \
        .applymap(style_last_sale_date, subset=['Last Sale Date']) \
        .applymap(style_last_unit_values, subset=['Last Unit Cost', 'Last Unit Price']) \
        .applymap(style_quote_counts, subset=['Pst 6M VQTs', 'Pst 6M CQTs'])

    st.subheader("Request for Quotes")
    # Display the dataframe without the index
    st.dataframe(styled_df, hide_index=True, height=600)

# --- VENDOR & CUSTOMER QUOTES --- #
elif page == "Vendor & Customer Quotes":
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
            # Define today's date for weight calculations
            today = pd.Timestamp.today()

            # Add weights based on recency
            filtered_vendor_df['Weight'] = (today - pd.to_datetime(filtered_vendor_df['ENTRY_DATE'])).dt.days.apply(
                lambda x: 1 / (x + 1))
            filtered_customer_df['Weight'] = (today - pd.to_datetime(filtered_customer_df['ENTRY_DATE'])).dt.days.apply(
                lambda x: 1 / (x + 1))

            # Calculate weighted averages
            avg_cost = (filtered_vendor_df['UNIT_COST'] * filtered_vendor_df['Weight']).sum() / filtered_vendor_df[
                'Weight'].sum() if not filtered_vendor_df.empty else None
            avg_price = (filtered_customer_df['UNIT_PRICE'] * filtered_customer_df['Weight']).sum() / \
                        filtered_customer_df['Weight'].sum() if not filtered_customer_df.empty else None

            # Calculate ranges and totals
            min_cost = filtered_vendor_df['UNIT_COST'].min() if not filtered_vendor_df.empty else None
            max_cost = filtered_vendor_df['UNIT_COST'].max() if not filtered_vendor_df.empty else None
            min_price = filtered_customer_df['UNIT_PRICE'].min() if not filtered_customer_df.empty else None
            max_price = filtered_customer_df['UNIT_PRICE'].max() if not filtered_customer_df.empty else None
            customer_qty = len(filtered_customer_df)
            vendor_qty = len(filtered_vendor_df)

            # Create a summary dataframe
            summary_df = pd.DataFrame({
                "Min Cost Range": [min_cost],
                "Max Cost Range": [max_cost],
                "Average Cost": [avg_cost],
                "Vendor Qty": [vendor_qty],
                "Min Price Range": [min_price],
                "Max Price Range": [max_price],
                "Average Price": [avg_price],
                "Customer Qty": [customer_qty]
            })

            # Display the summary dataframe
            st.subheader("Cost and Price Range Summary")
            st.dataframe(summary_df, width=1000)

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
                        marker=dict(color='#8FB8CA')
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
                        marker=dict(color='#91AC9A')
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

                    with st.expander("View All Vendor Quotes"):
                        filtered_vendor_df = filtered_vendor_df.drop(columns=["Weight", "PN", "DESCRIPTION"])
                        st.dataframe(filtered_vendor_df)

                with col2:
                    plot_quote_histogram(
                        filtered_customer_df,
                        part_number,
                        title="Frequency of Customer Quotes Over Time",
                        color='#91AC9A',
                        max_y_value=max_y_value
                    )

                    with st.expander("View All Customer Quotes"):
                        filtered_customer_df = filtered_customer_df.drop(columns=["Weight", "PN", "DESCRIPTION"])
                        st.dataframe(filtered_customer_df)
