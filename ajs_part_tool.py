# <editor-fold desc="# --- IMPORT MODULES & INITIAL PAGE CONFIG --- #">
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_navigation_bar import st_navbar

# Initial Page Configuration --- #
st.set_page_config(page_title="AJS Part Tool", page_icon="🔧", layout="wide")


# --- LOADING DATA FUNCTION --- #"
# Loading Data Function
@st.cache_data
def load_data():
    vendor_df = pd.read_excel('VQ Details Today_TR.xlsx')
    customer_df = pd.read_excel('CQ_Detail_TODAY_TR.xlsx')
    quote_df = pd.read_excel('quote_df.xlsx')
    pn_master_df = pd.read_excel('pn_master.xlsx')
    stock_df = pd.read_excel('stock_df_original.xlsx')
    sales_df = pd.read_excel('sales_df.xlsx')
    purchases_df = pd.read_excel('POs_LIVE_TR.xlsx')

    return vendor_df, customer_df, quote_df, pn_master_df, stock_df, sales_df, purchases_df


# Executing Loading Data Function
vendor_df, customer_df, quote_df, pn_master_df, stock_df, sales_df, purchases_df = load_data()


# --- ALL FUNCTIONS --- #"
# --- GET SALE DATA FUNCTIONS --- #
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

# --- FUNCTION & USES OF CONVERTING TO DATETIME --- #"
def convert_to_date(df, column):
    df[column] = pd.to_datetime(df[column]).dt.date

convert_to_date(vendor_df, 'ENTRY_DATE')
convert_to_date(customer_df, 'ENTRY_DATE')
convert_to_date(sales_df, 'ENTRY_DATE')
convert_to_date(purchases_df, 'ENTRY_DATE')

# --- PAGE SETUP --- #
# Navigation Bar
pages = ["Home", "Part Information", "Sales Data", "List Analysis"]
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


    @st.cache_data
    def filter_quote_data(customer_sort, part_sort):
        # Start with the full dataset
        filtered_df = quote_df.copy()

        # Apply filters
        if customer_sort:
            filtered_df = filtered_df[
                filtered_df['Company_Name'].str.contains(customer_sort, case=False, na=False)
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
        filtered_quote_df['QTY'] = filtered_quote_df['QTY'].astype(str)

        # Process rows with multiple part numbers
        split_rows = []
        qt_counter = 1  # Initialize QT# counter
        for _, row in filtered_quote_df.iterrows():
            row_df = pd.DataFrame([row])
            row_df['QT#'] = qt_counter  # Assign QT# to the row
            split_rows.append(row_df)
            qt_counter += 1  # Increment QT# for the next quote

        # Concatenate all rows back together
        processed_df = pd.concat(split_rows, ignore_index=True)

        # Add "Stock" column
        processed_df['Stock'] = processed_df['PN'].apply(lambda x: '' if x in stock_df['PN'].values else ' ')

        # Add "PN Master" column
        processed_df['PN Master'] = processed_df['PN'].apply(lambda x: '' if x in pn_master_df['PN'].values else ' ')
        processed_df['Last Sale Date'] = processed_df['PN'].apply(get_last_sale_date)
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
            'Last Unit Price': lambda x: f'{x:.2f}' if not pd.isna(x) and x != '' else ''
        })

        # Display the dataframe without additional styling
        st.dataframe(styled_df, hide_index=True, height=table_height)

    else:
        st.write("No matching data found.")

# --- PART INFO & ANALYSIS --- #
elif page == "Part Information":
    # Filter Out 0 Values
    vendor_df = vendor_df[vendor_df['UNIT_COST'] != 0]
    customer_df = customer_df[customer_df['UNIT_PRICE'] != 0]
    purchases_df = purchases_df[purchases_df['TOTAL_COST'] != 0]

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
        filtered_vendor_df = vendor_df[
            vendor_df['PN'].str.contains(part_number, case=False, na=False)
        ]
        filtered_customer_df = customer_df[
            customer_df['PN'].str.contains(part_number, case=False, na=False)
        ]
        filtered_sales_df = sales_df[
            sales_df['PN'].str.contains(part_number, case=False, na=False)
        ]
        filtered_purchases_df = purchases_df[
            purchases_df['PN'].str.contains(part_number, case=False, na=False)
        ]
        filtered_stock_df = stock_df[
            stock_df['PN'].str.contains(part_number, case=False, na=False)
        ]

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
                "Last Unit Price": [last_unit_price]
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
                st.dataframe(filtered_vendor_df.drop(columns=['PN', 'DESCRIPTION', 'Weight']), height=215, width=780,
                             hide_index=True, use_container_width=True)
                st.write("**Customer Data**")
                st.dataframe(filtered_customer_df.drop(columns=['PN', 'DESCRIPTION', 'Weight']), height=215, width=780,
                             hide_index=True, use_container_width=True)

            with col_sales_purchase:
                st.write("**Sales Data**")
                st.dataframe(filtered_sales_df.drop(columns=['PN', 'DESCRIPTION']), height=215, width=780,
                             hide_index=True, use_container_width=True)
                st.write("**Purchases Data**")
                st.dataframe(filtered_purchases_df.drop(columns=['PN', 'DESCRIPTION']), height=215, width=780,
                             hide_index=True, use_container_width=True)
            with col_stock:
                st.write("**Stock Data**")
                st.dataframe(filtered_stock_df.drop(columns=['PN', 'DESCRIPTION']), height=215, width=780,
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

# --- SALES DATA --- #
elif page == "Sales Data":
    st.subheader("Sale Data")
    part_number = st.text_input("Search for Part Number (PN)", "")

    # Only proceed if a part number is entered
    if part_number:
        # Apply filters to the sales data
        filtered_sales_df = sales_df[
            sales_df['PN'].str.contains(part_number, case=False, na=False)
        ]

        # Display the filtered sales data
        st.dataframe(filtered_sales_df)

# --- LIST ANALYSIS PAGE ---
elif page == "List Analysis":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("List Analysis")
        uploaded_file = st.file_uploader("Upload an Excel file for List Analysis", type=["xlsx"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        df['Last Sale Date'] = df['PN'].apply(get_last_sale_date)
        df['Last Unit Cost'] = df['PN'].apply(get_last_unit_cost)
        df['Last Unit Price'] = df['PN'].apply(get_last_unit_price)

        # Merge with sales_df to include "Transaction" column
        merged_df = pd.merge(df, sales_df[['PN', 'ROUTE_DESC']], on='PN', how='left')
        merged_df.rename(columns={'ROUTE_DESC': 'Transaction'}, inplace=True)

        # Initialize separate tables
        table_stock = pd.DataFrame(columns=merged_df.columns)
        table_sales = pd.DataFrame(columns=merged_df.columns)
        table_customer = pd.DataFrame(columns=merged_df.columns)
        table_no_sales_no_data = pd.DataFrame(columns=merged_df.columns)
        table_customer_price_zero = pd.DataFrame(columns=merged_df.columns)

        # Function to calculate weighted average, excluding exchanges
        def calculate_weighted_avg(part_number, column):
            """
            Calculate the weighted average for the given column in customer_df,
            excluding exchange quotes.
            """
            matching_data = customer_df[customer_df['PN'] == part_number].copy()

            if matching_data.empty:
                return 0, False  # Return 0 and indicate no valid data if empty

            # Ensure ENTRY_DATE is datetime
            matching_data['ENTRY_DATE'] = pd.to_datetime(matching_data['ENTRY_DATE'], errors='coerce')

            # Identify exchanges as rows where the price is significantly lower than the cost
            exchange_threshold = 0.5  # 50% difference
            matching_data = matching_data[
                ~(matching_data['UNIT_PRICE'] < matching_data['UNIT_COST'] * exchange_threshold)
            ]

            if matching_data.empty:
                return 0, True  # Return 0 and indicate only exchange data

            # Define weights based on recency
            today = pd.Timestamp.today()
            four_years_ago = today - pd.DateOffset(years=4)
            matching_data['Weight'] = (today - matching_data['ENTRY_DATE']).dt.days.apply(
                lambda x: 1 / max((x - (four_years_ago - today).days), 1)
            )

            # Calculate weighted average
            weighted_avg = (matching_data[column] * matching_data['Weight']).sum() / matching_data['Weight'].sum()
            return round(weighted_avg, 2), False

        # Separate rows into respective tables
        for _, row in merged_df.iterrows():
            if row['PN'] in stock_df['PN'].values:  # Check if part is in stock
                table_stock = pd.concat([table_stock, row.to_frame().T], ignore_index=True)
            elif pd.isna(row['Last Sale Date']):  # No sale data
                # Check if part exists in customer_df
                if row['PN'] in customer_df['PN'].values:
                    w_avg_price, only_exchange = calculate_weighted_avg(row['PN'], 'UNIT_PRICE')
                    w_avg_cost, _ = calculate_weighted_avg(row['PN'], 'UNIT_COST')
                    # If the weighted average unit price is 0 or only exchange data
                    if w_avg_price == 0 or only_exchange:
                        row['W Avg Price'] = w_avg_price
                        row['W Avg Cost'] = w_avg_cost
                        table_customer_price_zero = pd.concat([table_customer_price_zero, row.to_frame().T],
                                                              ignore_index=True)
                    else:  # Otherwise, add to the customer table
                        row['W Avg Price'] = w_avg_price
                        row['W Avg Cost'] = w_avg_cost
                        table_customer = pd.concat([table_customer, row.to_frame().T], ignore_index=True)
                elif row['PN'] not in vendor_df['PN'].values and row['PN'] not in customer_df['PN'].values:
                    table_no_sales_no_data = pd.concat([table_no_sales_no_data, row.to_frame().T], ignore_index=True)
            else:
                table_sales = pd.concat([table_sales, row.to_frame().T], ignore_index=True)

        with col1:
            st.write(f"Total Parts in List = {len(df)}")
            st.dataframe(df.drop(columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price']),
                         height=305, width=780, hide_index=True, use_container_width=True)

        # <editor-fold desc="# --- TABLES DISPLAY --- #">
        with col2:
            st.subheader(f"Data Used For Analysis = {len(table_sales) + len(table_customer)}")

            # Display Parts with Sale Data
            st.write(f"**Parts with Sale Data:** {len(table_sales)} Parts")
            table_sales = table_sales.drop(columns='Transaction')
            st.dataframe(table_sales, height=210, hide_index=True, use_container_width=True)  # Use container width

            # Display Parts with Customer Data
            st.write(f"**Parts with Customer Data:** {len(table_customer)} Parts")
            table_customer = table_customer.drop(
                columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price', 'Transaction'], errors='ignore')
            st.dataframe(table_customer, height=210, hide_index=True, use_container_width=True)  # Use container width
        with col3:
            st.subheader("Data Analysis")
            st.write("")
            # Calculate totals for table_sales
            total_unit_price = round(table_sales['Last Unit Price'].sum())
            total_unit_cost = round(table_sales['Last Unit Cost'].sum())

            # Calculate totals for table_customer (W Avg Price and W Avg Cost)
            total_w_avg_price = round(table_customer['W Avg Price'].sum())
            total_w_avg_cost = round(table_customer['W Avg Cost'].sum())

            potential_profit = (total_unit_price - total_unit_cost) + (total_w_avg_price - total_w_avg_cost)

            st.write("**Parts with Sale Data**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Sum Unit Price:")
                st.write(f"Sum Unit Cost:")
            with col2:
                st.write(f"{total_unit_price}")
                st.write(f"{total_unit_cost}")
            st.markdown("---")

            st.write("**Parts with Customer Data**")
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
            with col2:
                st.write(f"{potential_profit}")

        st.markdown("---")

        st.subheader(
            f"Data Not Used For Analysis = "
            f"{len(table_customer_price_zero) + len(table_stock) + len(table_no_sales_no_data)}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(
                f"**PN w/o Sale Data & w/ Customer Data w/ Unit Price = 0**: "
                f"{len(table_customer_price_zero)} Parts")
            table_customer_price_zero = table_customer_price_zero.drop(
                columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price', 'Transaction'], errors='ignore')
            st.dataframe(table_customer_price_zero, height=420, width=780,
                         hide_index=True, use_container_width=True)
        with col2:
            st.write(
                f"**PN w/o Sale Data or Customer/Vendor Data: {len(table_no_sales_no_data)} Parts**")
            table_no_sales_no_data = table_no_sales_no_data.drop(
                columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price', 'Transaction'], errors='ignore')
            st.dataframe(table_no_sales_no_data, height=420, width=780,
                         hide_index=True, use_container_width=True)
        with col3:
            st.write(f"**PN Already in Stock: {len(table_stock)} Parts**")
            table_stock = table_stock.drop(
                columns=['Last Sale Date', 'Last Unit Cost', 'Last Unit Price', 'Transaction'], errors='ignore')
            st.dataframe(table_stock, height=420, width=780,
                         hide_index=True, use_container_width=True)
        # </editor-fold>
