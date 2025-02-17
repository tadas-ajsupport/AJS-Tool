# <editor-fold desc="# --- IMPORT MODULES & INITIAL PAGE CONFIG --- #">
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_navigation_bar import st_navbar
from datetime import date
import re

# Initial Page Configuration --- #
st.set_page_config(page_title="AJS Part Tool", page_icon="🔧", layout="wide")
today = date.today()

# </editor-fold>

# <editor-fold desc="# --- LOADING DATA FUNCTION --- #">
# Loading Data Function
def load_quote_data(file_path):
    return pd.read_excel(file_path)

quote_df = load_quote_data('email_scrape_results.xlsx')
vq_details = load_quote_data('Scraping Results.xlsx')

@st.cache_data
def load_data():
    vendor_df = pd.read_excel('VQ Details Today_TR.xlsx')
    customer_df = pd.read_excel('CQ_Detail_TODAY_TR.xlsx')
    pn_master_df = pd.read_excel('pn_master.xlsx')
    stock_df = pd.read_excel('STOCK_241017.xlsx')
    sales_df = pd.read_excel('sales_df.xlsx')
    purchases_df = pd.read_excel('POs_LIVE_TR.xlsx')
    activity_df = pd.read_excel('parts_activity.xlsx')

    return vendor_df, customer_df, pn_master_df, stock_df, sales_df, purchases_df, activity_df


# Executing Loading Data Function
vendor_df, customer_df, pn_master_df, stock_df, sales_df, purchases_df, activity_df = load_data()


# </editor-fold>

# <editor-fold desc="# --- ALL FUNCTIONS --- #">
# <editor-fold desc="# --- GET SALE DATA FUNCTIONS --- #>
# Add "Last Sale Date" column
def get_last_sale_date(part_number):
    # Filter sales_df for rows matching the part number
    matching_sales = sales_df[sales_df['PN'] == part_number]
    if not matching_sales.empty:
        return matching_sales['ENTRY_DATE'].max()  # Return the most recent sale date
    return None  # No sales found


# Add "Last Unit Cost" and "Last Unit Price" columns
def get_last_unit_cost(part_number):
    # Filter sales_df for rows matching the part number
    matching_sales = sales_df[sales_df['PN'] == part_number]
    if not matching_sales.empty:
        return round(matching_sales.loc[matching_sales['ENTRY_DATE'].idxmax(), 'UNIT_COST'],
                     2)  # Get UNIT_COST for the most recent sale and round
    return ''  # No sales found


def get_last_unit_price(part_number):
    # Filter sales_df for rows matching the part number
    matching_sales = sales_df[sales_df['PN'] == part_number]
    if not matching_sales.empty:
        return round(matching_sales.loc[matching_sales['ENTRY_DATE'].idxmax(), 'UNIT_PRICE'],
                     2)  # Get UNIT_PRICE for the most recent sale and round
    return ''  # No sales found


# </editor-fold>

# --- HISTOGRAM GENERATION --- #
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


# </editor-fold>

# <editor-fold desc="# --- FUNCTION & USES OF CONVERTING TO DATETIME --- #">
def convert_to_date(df, column):
    df[column] = pd.to_datetime(df[column]).dt.date


convert_to_date(vendor_df, 'ENTRY_DATE')
convert_to_date(customer_df, 'ENTRY_DATE')
convert_to_date(sales_df, 'ENTRY_DATE')
convert_to_date(purchases_df, 'ENTRY_DATE')
convert_to_date(vq_details, 'Timestamp')
# convert_to_date(stock_df, 'ENTRY_DATE')
quote_df['Timestamp'] = pd.to_datetime(quote_df['Timestamp'])
quote_df['Timestamp'] = quote_df['Timestamp'].dt.date
# </editor-fold>

# <editor-fold desc="# --- PAGE SETUP --- #">
# Navigation Bar
pages = ["Home", "Part Information", "List Analysis", "Vendor Quotes"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "Untitled design.svg")  # Update logo path

# Styling
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
# </editor-fold>

# --- HOME PAGE ---
if page == "Home":

    st.subheader("Request for Quotes")

    # Filter input fields
    customer_col, part_col, empty_col_1, empty_col_2 = st.columns(4)
    with customer_col:
        customer_sort = st.text_input("Filter by Company Name", "")
    with part_col:
        part_sort = st.text_input("Filter by PN", "")


    def filter_quote_data(customer_sort, part_sort):
        # Start with the full dataset
        filtered_df = quote_df

        # Apply filters
        if customer_sort:
            filtered_df = filtered_df[
                filtered_df['Company Name'].str.contains(customer_sort, case=False, na=False)
            ]
        if part_sort:
            filtered_df = filtered_df[
                filtered_df['PN'].str.contains(part_sort, case=False, na=False)
            ]

        return filtered_df


    # Apply filters with caching
    filtered_quote_df = filter_quote_data(customer_sort, part_sort)

    # Check if any data remains after filtering
    if not filtered_quote_df.empty:

        # Ensure 'PN' and 'QTY' columns are of type string for processing
        filtered_quote_df['PN'] = filtered_quote_df['PN'].astype(str)
        filtered_quote_df['Quantity'] = filtered_quote_df['Quantity'].astype(str)

        # Process rows with multiple part numbers and descriptions
        split_rows = []
        qt_counter = 1  # Initialize QT# counter

        for _, row in filtered_quote_df.iterrows():
            part_numbers = row['PN'].split(';')  # Split part numbers by semicolon
            descriptions = row['Description'].split(';') if ';' in str(row['Description']) else [row[
                                                                                                     'Description']] * len(
                part_numbers)

            # Ensure descriptions match the number of part numbers
            descriptions = [desc.strip() for desc in descriptions]  # Clean whitespace
            if len(descriptions) < len(part_numbers):
                descriptions += [''] * (
                            len(part_numbers) - len(descriptions))  # Fill missing descriptions with empty strings

            for part_number, description in zip(part_numbers, descriptions):
                part_number = part_number.strip()  # Remove any leading/trailing spaces
                new_row = row.copy()  # Copy the original row
                new_row['PN'] = part_number  # Assign the current part number
                new_row['Description'] = description  # Assign the corresponding description
                new_row['QT#'] = qt_counter  # Assign the same QT#
                split_rows.append(new_row)  # Append the new row to the list
            qt_counter += 1  # Increment QT# for the next group of quotes

        # Concatenate all rows back together
        processed_df = pd.DataFrame(split_rows)

        # Add "Stock" column
        processed_df['Stock'] = processed_df['PN'].apply(lambda x: '' if x in stock_df['PN'].values else ' ')

        # Add "PN Master" column
        processed_df['PN Master'] = processed_df['PN'].apply(lambda x: '' if x in pn_master_df['PN'].values else ' ')
        processed_df['Last Sale Date'] = processed_df['PN'].apply(get_last_sale_date)
        processed_df['Last Unit Cost'] = processed_df['PN'].apply(get_last_unit_cost)
        processed_df['Last Unit Price'] = processed_df['PN'].apply(get_last_unit_price)

        # Merge with activity_df to add the Activity Score
        processed_df = pd.merge(processed_df, activity_df[['PN', 'Activity Score']], on='PN', how='left')

        # Replace None or NaN with an empty string in the Activity Score column
        processed_df['Activity Score'] = processed_df['Activity Score'].fillna('')

        # Replace None or NaN with an empty string in other relevant columns
        processed_df[['Last Unit Cost', 'Last Unit Price', 'Last Sale Date']] = processed_df[
            ['Last Unit Cost', 'Last Unit Price', 'Last Sale Date']].fillna('')

        # Ensure numeric columns remain numeric and empty cells for invalid values
        processed_df[['Last Unit Cost', 'Last Unit Price']] = processed_df[['Last Unit Cost', 'Last Unit Price']].apply(
            pd.to_numeric, errors='coerce'
        ).fillna('')

        # Move 'QT#' column to the first position
        columns = ['QT#'] + [col for col in processed_df.columns if col != 'QT#']
        processed_df = processed_df[columns]

        # Styling functions
        def color_cells(value):
            if value == '':
                return 'background-color: #7ABD7E; color: white;'  # Green
            elif value == ' ':
                return 'background-color: #FF6961; color: white;'  # Red
            return ''

        # Dynamic height adjustment
        num_rows = len(processed_df)
        table_height = min(40 + num_rows * 35, 800)  # Dynamic height, capped at 800px

        # Display the dataframe with styling
        styled_df = processed_df.style \
            .applymap(color_cells, subset=['Stock', 'PN Master']) \
            .format({
            'Last Unit Cost': lambda x: f'{x:.2f}' if not pd.isna(x) and x != '' else '',
            'Last Unit Price': lambda x: f'{x:.2f}' if not pd.isna(x) and x != '' else '',
            'Activity Score': lambda x: f'{x:.2f}' if isinstance(x, (float, int)) else ''  # Format numeric scores
        })

        # Display the dataframe without additional styling
        st.dataframe(styled_df, hide_index=True, height=table_height - 5)

    else:
        st.write(len(quote_df))
        st.write("No matching data found.")

# --- PART INFO & ANALYSIS --- #
elif page == "Part Information":
    # Filter Out 0 Values
    vendor_df = vendor_df[vendor_df['UNIT_COST'] != 0]
    customer_df = customer_df[customer_df['UNIT_PRICE'] != 0]
    purchases_df = purchases_df[purchases_df['TOTAL_COST'] != 0]
    sales_df = sales_df[sales_df['COST'] != 0]

    st.subheader("Vendor & Customer Quotes")

    # Add search box for part number and condition code filter
    col_part_search, col_filters = st.columns(2)

    with col_part_search:
        part_number = st.text_input("Search for Part Number (PN)", "")
    with col_filters:
        col_cond_filter, con_yr_filter = st.columns(2)

        with col_cond_filter:
            condition_code = st.text_input("Filter by Condition Code (e.g., NE, FN, OH)", "")

        with con_yr_filter:
            year_filter = st.text_input("Filter by Year (e.g., 2020)", "")
            year_filter = int(year_filter) if year_filter.isdigit() else None

    # Only proceed if a part number is entered
    if part_number:
        # <editor-fold desc="# --- FILTERS --- #">
        # Apply filters to the data
        filtered_vendor_df = vendor_df[vendor_df['PN'].str.strip().str.upper() == part_number.strip().upper()]
        filtered_customer_df = customer_df[customer_df['PN'].str.strip().str.upper() == part_number.strip().upper()]
        filtered_sales_df = sales_df[sales_df['PN'].str.strip().str.upper() == part_number.strip().upper()]
        filtered_purchases_df = purchases_df[purchases_df['PN'].str.strip().str.upper() == part_number.strip().upper()]
        filtered_stock_df = stock_df[stock_df['PN'].str.strip().str.upper() == part_number.strip().upper()]

        if condition_code:
            filtered_vendor_df = filtered_vendor_df[
                filtered_vendor_df['CONDITION_CODE'].str.contains(condition_code, case=False, na=False)
            ]
            filtered_customer_df = filtered_customer_df[
                filtered_customer_df['CONDITION_CODE'].str.contains(condition_code, case=False, na=False)
            ]
            filtered_sales_df = filtered_sales_df[
                filtered_sales_df['CONDITION_CODE'].str.contains(condition_code, case=False, na=False)
            ]
            filtered_purchases_df = filtered_purchases_df[
                filtered_purchases_df['CONDITION_CODE'].str.contains(condition_code, case=False, na=False)
            ]
            filtered_stock_df = stock_df[
                stock_df['CONDITION_CODE'].str.contains(part_number, case=False, na=False)
            ]

        if year_filter:
            filtered_vendor_df = filtered_vendor_df[
                pd.to_datetime(filtered_vendor_df['ENTRY_DATE']).dt.year >= year_filter
                ]
            filtered_customer_df = filtered_customer_df[
                pd.to_datetime(filtered_customer_df['ENTRY_DATE']).dt.year >= year_filter
                ]
            filtered_sales_df = filtered_sales_df[
                pd.to_datetime(filtered_sales_df['ENTRY_DATE']).dt.year >= year_filter
                ]
            filtered_purchases_df = filtered_purchases_df[
                pd.to_datetime(filtered_purchases_df['ENTRY_DATE']).dt.year >= year_filter
                ]
            filtered_stock_df = stock_df[
                stock_df['ENTRY_DATE'].str.contains(part_number, case=False, na=False)
            ]
        # </editor-fold>

        # <editor-fold desc="# --- COST & PRICE RANGE SUMMARY --- #">
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

            # Retrieve sales data for the searched part number
            last_sale_date = get_last_sale_date(part_number)
            last_unit_cost = get_last_unit_cost(part_number)
            last_unit_price = get_last_unit_price(part_number)

            description = pn_master_df.loc[pn_master_df['PN'] == part_number, 'DESCRIPTION'].values[0] \
                if part_number in pn_master_df['PN'].values else "No description available"

            # Get the activity score for the current part number
            activity_score = activity_df.loc[activity_df['PN'] == part_number, 'Activity Score'].values
            activity_score = activity_score[0] if len(activity_score) > 0 else None

            # Update the summary dataframe to include sales information
            summary_df = pd.DataFrame({
                "Description": [description],
                "Min Cost": [min_cost],
                "Max Cost": [max_cost],
                "Avrg Cost": [avg_cost],
                "Vendor Qty": [vendor_qty],
                "Min Price": [min_price],
                "Max Price": [max_price],
                "Avrg Price": [avg_price],
                "Customer Qty": [customer_qty],
                "Last Sale Date": [last_sale_date],
                "Last Unit Cost": [last_unit_cost],
                "Last Unit Price": [last_unit_price],
                "Activity Score": [activity_score]
            })

            # Display the summary dataframe
            st.write("**Cost and Price Range Summary**")
            st.dataframe(summary_df, hide_index=True, width=1400)
        # </editor-fold>

        tab_part_info, tab_part_analysis = st.tabs(['Part Information', 'Part Analysis'])

        # --- PART INFO TAB DISPLAY --- #
        with tab_part_info:
            col_vendor_customer, col_sales_purchase, col_stock = st.columns(3)

            with col_vendor_customer:
                st.write("**Vendor Data**")
                st.dataframe(filtered_vendor_df.drop(columns=['PN', 'DESCRIPTION']), height=210, width=780,
                             hide_index=True, use_container_width=True)
                st.write("**Customer Data**")
                st.dataframe(filtered_customer_df.drop(columns=['PN', 'DESCRIPTION']), height=210, width=780,
                             hide_index=True, use_container_width=True)

            with col_sales_purchase:
                st.write("**Sales Data**")
                st.dataframe(filtered_sales_df.drop(columns=['PN', 'DESCRIPTION']), height=210, width=780,
                             hide_index=True, use_container_width=True)
                st.write("**Purchases Data**")
                st.dataframe(filtered_purchases_df.drop(columns=['PN', 'DESCRIPTION']), height=210, width=780,
                             hide_index=True, use_container_width=True)
            with col_stock:
                st.write("**Stock Data**")
                st.dataframe(filtered_stock_df.drop(columns=['PN', 'DESCRIPTION']), height=210, width=780,
                             hide_index=True, use_container_width=True)

        # Add content to Tab 2
        with tab_part_analysis:
            # Create two columns for the box plots
            col_cost_range, col_cost_hist, col_price_range, col_price_hist = st.columns(4)

            # Get the maximum y-value for synchronized scaling
            max_y_value = get_max_histogram_value(filtered_vendor_df, filtered_customer_df, part_number)

            with col_cost_range:
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

            with col_price_range:
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

            with col_cost_hist:
                plot_quote_histogram(
                    filtered_vendor_df,
                    part_number,
                    title="Frequency of Vendor Quotes Over Time",
                    color='#8FB8CA',
                    max_y_value=max_y_value
                )

            with col_price_hist:
                plot_quote_histogram(
                    filtered_customer_df,
                    part_number,
                    title="Frequency of Customer Quotes Over Time",
                    color='#91AC9A',
                    max_y_value=max_y_value
                )

# --- LIST ANALYSIS PAGE ---
elif page == "List Analysis":

    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = None

    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("List Analysis")

        # File uploader
        uploaded_file = st.file_uploader("Upload an Excel file for List Analysis", type=["xlsx"])

        # If a new file is uploaded, store it in session state and reset the analysis
        if uploaded_file:
            if st.session_state["uploaded_file"] != uploaded_file:
                st.session_state["uploaded_file"] = uploaded_file
                st.session_state["analysis_results"] = None  # Reset analysis results for new file

        # Check if a file is present in session state
        if st.session_state["uploaded_file"]:
            if st.session_state["analysis_results"] is None:
                # Initialize empty tables to avoid NameErrors
                table_customer_price_zero = pd.DataFrame()
                table_stock = pd.DataFrame()
                table_sales = pd.DataFrame()
                table_customer = pd.DataFrame()
                table_no_sales_no_data = pd.DataFrame()

                # Perform the analysis only if it hasn't been done already
                try:
                    df = pd.read_excel(st.session_state["uploaded_file"], engine="openpyxl")
                except Exception as e:
                    st.error(f"An error occurred while reading the uploaded file: {e}")
                    st.stop()

                if 'SN' in df.columns:
                    has_sn_column = True
                else:
                    has_sn_column = False

                # Perform the analysis
                df = pd.merge(df, activity_df[['PN', 'Activity Score']], on='PN', how='left')
                df['Last Sale Date'] = df['PN'].apply(get_last_sale_date)
                df['Last Unit Cost'] = df['PN'].apply(get_last_unit_cost)
                df['Last Unit Price'] = df['PN'].apply(get_last_unit_price)

                sales_df['ROUTE_DESC'] = sales_df['ROUTE_DESC'].fillna("Unknown")  # Handle missing transaction types
                merged_df = pd.merge(df, sales_df[['PN', 'ROUTE_DESC']], on='PN', how='left')
                merged_df.rename(columns={'ROUTE_DESC': 'Transaction'}, inplace=True)

                # Initialize and populate tables
                table_stock = pd.DataFrame(columns=merged_df.columns)
                table_sales = pd.DataFrame(columns=merged_df.columns)
                table_customer = pd.DataFrame(columns=merged_df.columns)
                table_no_sales_no_data = pd.DataFrame(columns=merged_df.columns)
                table_customer_price_zero = pd.DataFrame(columns=merged_df.columns)

                # Reorder columns to make "Activity Score" the first column
                table_stock = table_stock[
                    ["Activity Score"] + [col for col in table_stock.columns if col != "Activity Score"]]
                table_sales = table_sales[
                    ["Activity Score"] + [col for col in table_sales.columns if col != "Activity Score"]]
                table_customer = table_customer[
                    ["Activity Score"] + [col for col in table_customer.columns if col != "Activity Score"]]
                table_no_sales_no_data = table_no_sales_no_data[
                    ["Activity Score"] + [col for col in table_no_sales_no_data.columns if col != "Activity Score"]]
                table_customer_price_zero = table_customer_price_zero[
                    ["Activity Score"] + [col for col in table_customer_price_zero.columns if col != "Activity Score"]]

                def calculate_weighted_avg(part_number, column):
                    matching_data = customer_df[customer_df['PN'] == part_number].copy()
                    if matching_data.empty:
                        return 0, False

                    matching_data['ENTRY_DATE'] = pd.to_datetime(matching_data['ENTRY_DATE'], errors='coerce')
                    exchange_threshold = 0.5
                    matching_data = matching_data[
                        ~(matching_data['UNIT_PRICE'] < matching_data['UNIT_COST'] * exchange_threshold)
                    ]

                    if matching_data.empty:
                        return 0, True

                    today = pd.Timestamp.today()
                    four_years_ago = today - pd.DateOffset(years=4)
                    matching_data['Weight'] = (today - matching_data['ENTRY_DATE']).dt.days.apply(
                        lambda x: 1 / max((x - (four_years_ago - today).days), 1)
                    )

                    weighted_avg = (matching_data[column] * matching_data['Weight']).sum() / matching_data['Weight'].sum()
                    return round(weighted_avg, 2), False

                for _, row in merged_df.iterrows():
                    if row['PN'] in stock_df['PN'].values:
                        table_stock = pd.concat([table_stock, row.to_frame().T], ignore_index=True)
                    elif pd.isna(row['Last Sale Date']):
                        if row['PN'] in customer_df['PN'].values:
                            w_avg_price, only_exchange = calculate_weighted_avg(row['PN'], 'UNIT_PRICE')
                            w_avg_cost, _ = calculate_weighted_avg(row['PN'], 'UNIT_COST')
                            if w_avg_price == 0 or only_exchange:
                                row['W Avg Price'] = w_avg_price
                                row['W Avg Cost'] = w_avg_cost
                                table_customer_price_zero = pd.concat([table_customer_price_zero, row.to_frame().T],
                                                                      ignore_index=True)
                            else:
                                row['W Avg Price'] = w_avg_price
                                row['W Avg Cost'] = w_avg_cost
                                table_customer = pd.concat([table_customer, row.to_frame().T], ignore_index=True)
                        elif row['PN'] not in vendor_df['PN'].values and row['PN'] not in customer_df['PN'].values:
                            table_no_sales_no_data = pd.concat([table_no_sales_no_data, row.to_frame().T], ignore_index=True)
                    else:
                        table_sales = pd.concat([table_sales, row.to_frame().T], ignore_index=True)

                st.session_state["analysis_results"] = {
                    "df": df,
                    "table_stock": table_stock,
                    "table_sales": table_sales,
                    "table_customer": table_customer,
                    "table_no_sales_no_data": table_no_sales_no_data,
                    "table_customer_price_zero": table_customer_price_zero,
                }

            analysis_results = st.session_state["analysis_results"]
            df = analysis_results["df"]
            table_stock = analysis_results["table_stock"]
            table_sales = analysis_results["table_sales"]
            table_customer = analysis_results["table_customer"]
            table_no_sales_no_data = analysis_results["table_no_sales_no_data"]
            table_customer_price_zero = analysis_results["table_customer_price_zero"]

            if has_sn_column:
                table_stock = table_stock.drop_duplicates(subset=['PN', 'SN'])
                table_sales = table_sales.drop_duplicates(subset=['PN', 'SN'])
                table_customer = table_customer.drop_duplicates(subset=['PN', 'SN'])
                table_no_sales_no_data = table_no_sales_no_data.drop_duplicates(subset=['PN', 'SN'])
                table_customer_price_zero = table_customer_price_zero.drop_duplicates(subset=['PN', 'SN'])
            else:
                table_stock = table_stock.drop_duplicates(subset=['PN'])
                table_sales = table_sales.drop_duplicates(subset=['PN'])
                table_customer = table_customer.drop_duplicates(subset=['PN'])
                table_no_sales_no_data = table_no_sales_no_data.drop_duplicates(subset=['PN'])
                table_customer_price_zero = table_customer_price_zero.drop_duplicates(subset=['PN'])

            # Replace negative values in Last Unit Price and Last Unit Cost with 0
            table_sales.loc[table_sales['Last Unit Price'] < 0, 'Last Unit Price'] = 0
            table_sales.loc[table_sales['Last Unit Cost'] < 0, 'Last Unit Cost'] = 0

            with col1:
                st.write(f"Total Parts in List = {len(df)}")
                st.dataframe(df.drop(columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price']),
                             height=305, width=780, hide_index=True, use_container_width=True)

            with col2:
                st.subheader(f"Data Used For Analysis = {len(table_sales) + len(table_customer)}")

                st.write(f"**with SO Data:** {len(table_sales)} Parts")
                table_sales = table_sales.drop(columns='Transaction')
                st.dataframe(table_sales, height=210, width=780, hide_index=True, use_container_width=True)

                st.write(f"**with CQ Data Only:** {len(table_customer)} Parts")
                table_customer = table_customer.drop(
                    columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price', 'Transaction'], errors='ignore')
                st.dataframe(table_customer, height=210, width=780, hide_index=True, use_container_width=True)

            with col3:
                # Detect the quantity column from the uploaded file
                quantity_column = next(
                    (col for col in ['QTY', 'Qty', 'quantity', 'Quantity'] if col in table_sales.columns),
                    None  # Default to None if no column is found
                )

                # Clean and validate the quantity column
                if quantity_column:
                    # Extract numeric values from the quantity column
                    table_sales[quantity_column] = (
                        table_sales[quantity_column]
                        .astype(str)  # Ensure the column is treated as strings
                        .str.extract(r'(\d+)')  # Extract numeric values as strings
                        .fillna(0)  # Replace missing values with 0
                        .astype(int)  # Convert to integers
                    )
                else:
                    # Set a default quantity if no column is found
                    st.warning("No quantity column found in the uploaded file. Defaulting all quantities to 1.")
                    table_sales['Quantity'] = 1
                    quantity_column = 'Quantity'

                st.subheader("Data Analysis")
                st.write("")

                # Calculate total unit price and cost based on quantity
                table_sales['Total Price'] = table_sales['Last Unit Price'] * table_sales[quantity_column]
                table_sales['Total Cost'] = table_sales['Last Unit Cost'] * table_sales[quantity_column]

                total_unit_price = round(table_sales['Total Price'].sum())
                total_unit_cost = round(table_sales['Total Cost'].sum())

                # Check if the table is not empty and contains the required columns
                if not table_customer.empty and 'W Avg Price' in table_customer.columns:
                    total_w_avg_price = round(table_customer['W Avg Price'].sum())
                else:
                    total_w_avg_price = 0  # Default value when table is empty or column is missing

                if not table_customer.empty and 'W Avg Cost' in table_customer.columns:
                    total_w_avg_cost = round(table_customer['W Avg Cost'].sum())
                else:
                    total_w_avg_cost = 0  # Default value when table is empty or column is missing

                potential_profit = (total_unit_price - total_unit_cost) + (total_w_avg_price - total_w_avg_cost)

                st.write("**with SO Data**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Sum Unit Price:")
                    st.write(f"Sum Unit Cost:")
                with col2:
                    st.write(f"{total_unit_price}")
                    st.write(f"{total_unit_cost}")
                st.markdown("---")

                st.write("**with CQ Data Only**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Sum Unit Price:")
                    st.write(f"Sum Unit Cost:")
                with col2:
                    st.write(f"{total_w_avg_price}")
                    st.write(f"{total_w_avg_cost}")
                st.markdown("---")

                st.write("**Potential Profit**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Potential Profit Range:")
                    st.write("List's Activity Score:")
                with col2:
                    st.write(f"{potential_profit}")

                    # Replace activity scores "<1" with a numeric value of 0.5 for proper calculation
                    def preprocess_activity_scores(df, column):
                        df[column] = df[column].apply(
                            lambda x: 0.5 if isinstance(x, str) and x.strip() == "<1" else x
                        )
                        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)

                    # Apply preprocessing to ensure all activity scores are numeric
                    preprocess_activity_scores(table_sales, 'Activity Score')
                    preprocess_activity_scores(table_customer, 'Activity Score')

                    # Calculate total activity score and count
                    total_activity_sum = table_sales['Activity Score'].sum() + table_customer['Activity Score'].sum()
                    total_count = len(table_sales['Activity Score']) + len(table_customer['Activity Score'])

                    # Prevent division by zero
                    if total_count > 0:
                        average_activity_score = round(total_activity_sum / total_count, 2)
                    else:
                        average_activity_score = 0  # Default value when there are no activity scores

                    st.write(f"{average_activity_score}")

        else:
            # Ensure empty tables are initialized to avoid errors when displaying results
            table_customer_price_zero = pd.DataFrame()
            table_stock = pd.DataFrame()
            table_sales = pd.DataFrame()
            table_customer = pd.DataFrame()
            table_no_sales_no_data = pd.DataFrame()

            st.warning("Please upload an Excel file to perform the analysis.")

    st.markdown("---")
    st.subheader(
        f"Data Not Used For Analysis = "
        f"{len(table_customer_price_zero) + len(table_stock) + len(table_no_sales_no_data)}"
    )

    # Create three columns for the tables
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(
            f"**with CQ Data only of Unit Price = 0**: "
            f"{len(table_customer_price_zero)} Parts"
        )
        table_customer_price_zero = table_customer_price_zero.drop(
            columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price', 'Transaction'], errors='ignore')
        st.dataframe(table_customer_price_zero, height=420, hide_index=True)

    with col2:
        st.write(
            f"**without SO or CQ/VQ Data: {len(table_no_sales_no_data)} Parts**"
        )
        table_no_sales_no_data = table_no_sales_no_data.drop(
            columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price', 'Transaction'], errors='ignore')
        st.dataframe(table_no_sales_no_data, height=420, hide_index=True)

    with col3:
        st.write(f"**Already in Stock: {len(table_stock)} Parts**")
        table_stock = table_stock.drop(
            columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price', 'Transaction'], errors='ignore')
        st.dataframe(table_stock, height=420, hide_index=True)

# --- VENDOR QUOTE PAGE --- #
elif page == "Vendor Quotes":
    st.subheader(f"Vendor Quote Directory - {today}")

    # Ensure the main DataFrame is stored in session state for persistence
    if "vq_details" not in st.session_state:
        st.session_state.vq_details = vq_details.copy()  # Store DataFrame in session state

    # Loading Manual Score
    score_df = pd.read_excel("sucess_score.xlsx")

    # Strip spaces from column names to ensure they match exactly
    vq_details.columns = vq_details.columns.str.strip()
    score_df.columns = score_df.columns.str.strip()

    # Check if "VQ#" and "Score" exist in both DataFrames
    if "VQ#" in vq_details.columns and "VQ#" in score_df.columns and "Score" in score_df.columns:
        # Merge on "VQ#" instead of "PN"
        vq_details = vq_details.merge(score_df[["VQ#", "Score"]], on="VQ#", how="left")

        # Fill missing scores with 0
        vq_details["Score"] = vq_details["Score"].fillna(0)
    else:
        st.error("Column mismatch: Ensure 'VQ#' and 'Score' exist in both data sources.")

    vq_details["Score"] = vq_details["Score"].fillna(0)

    # Function to expand rows based on "PN" column while maintaining column integrity
    def expand_vendor_quote_data(df):
        if "PN" not in df.columns:
            return df  # Return original if "PN" column is missing

        expanded_rows = []

        for _, row in df.iterrows():
            pn_values = str(row["PN"]).split("; ") if pd.notna(row["PN"]) and "; " in str(row["PN"]) else [row["PN"]]

            # Identify other columns that might have multiple values
            split_values = {
                col: str(row[col]).split("; ") if pd.notna(row[col]) and "; " in str(row[col]) else [row[col]]
                for col in df.columns if col != "PN"
            }

            max_length = max(len(pn_values), *[len(v) for v in split_values.values()])

            # Ensure all columns have consistent row counts
            for col in split_values:
                if len(split_values[col]) == 1:
                    split_values[col] *= max_length  # Repeat single values across new rows
                else:
                    split_values[col] += [""] * (max_length - len(split_values[col]))  # Pad shorter lists

            # Create new rows
            for i in range(len(pn_values)):
                new_row = {col: split_values[col][i] if col != "PN" else pn_values[i] for col in df.columns}
                expanded_rows.append(new_row)

        return pd.DataFrame(expanded_rows)

    # Filter search bar
    pn_sort_col, subj_sort_col = st.columns(2)
    with pn_sort_col:
        pn_filter = st.text_input("Filter by PN", "")
    with subj_sort_col:
        subj_filter = st.text_input("Filter by Subject", "")

    # Define the filtering logic
    def filter_quote_data(pn_filter, subj_filter):
        filtered_df = vq_details.copy()

        if pn_filter:
            filtered_df = filtered_df[
                filtered_df["PN"].astype(str).str.contains(pn_filter, case=False, na=False)
            ]

        if subj_filter:
            filtered_df = filtered_df[
                filtered_df["Subject"].astype(str).str.contains(subj_filter, case=False, na=False)
            ]

        st.session_state.vq_details = filtered_df  # Update session state with filtered data

    # Apply the filter and update the session state
    filter_quote_data(pn_filter, subj_filter)

    # UI toggle for row expansion
    expand_rows = st.checkbox("Expand Rows")

    # Apply expansion logic if checkbox is selected
    if expand_rows:
        st.session_state.vq_details = expand_vendor_quote_data(st.session_state.vq_details)

    # Display the DataFrame
    if not st.session_state.vq_details.empty:
        st.dataframe(st.session_state.vq_details, hide_index=True)
    else:
        st.warning("No data matches the filter.")





