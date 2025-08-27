import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from folium.plugins import FeatureGroupSubGroup
import re

# Set page config
st.set_page_config(page_title="OTR App", 
                   layout="wide",
                   page_icon="🔓",
                   initial_sidebar_state="expanded")

# Add custom CSS to reduce gap between top of page and title
st.markdown("""
<style>
    .block-container {
        padding-top: 0.5rem !important;
        margin-top: 0.5rem !important;
    }
    .main > div {
        padding-top: 0.5rem !important;
    }
    div[data-testid="stAppViewContainer"] > .main {
        padding-top: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Hide the main menu (hamburger menu)
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Hide the footer
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Hide the header (including GitHub, Share, etc.) - Modified to not affect sidebar
st.markdown("""
<style>
header[data-testid="stHeader"] {visibility: hidden;}
/* Ensure sidebar is always visible */
.css-1d391kg {visibility: visible !important;}
section[data-testid="stSidebar"] {visibility: visible !important;}
</style>
""", unsafe_allow_html=True)

# Constants
REQUIRED_OTR_COLUMNS = ['Asset Name', 'SSDB_REF', 'Area unit','Currency Unit', 'Valuation date', 'MLA', 'CLA',
                        'SS_CLA', 'SS_MLA', 'Occ area', 'Occ % CLA', 'Current Rent',
                        'Anc Inc','Retail', 'Other', 'Insurance',
                        'Staff', 'Marketing', 'Utilities', 'Rates', 'Rent',
                        'SS Unit Count', 'Avg SS Unit Size', 'SS Revenue', 'Total Rev',
                        'Opex Total', 'EBITDAR', 'EBITDA']

OTR_STRING_COLUMNS = ['Asset Name', 'SSDB_REF', 'Currency Unit']
OTR_ROUNDED_COLUMNS = ['Staff', 'Marketing', 'Utilities', 'Rates', 'Rent']
OTR_PERCENTAGE_COLUMNS = ['Anc Inc', 'Retail', 'Other', 'Insurance','Occ % CLA']
OTR_NUMERIC_COLUMNS = ['Anc Inc', 'Retail', 'Other', 'Insurance', 'SS_CLA']
OTR_FLOAT_COLUMNS = ['Current Rent']

SSDB_REQUIRED_COLUMNS = ['SSDB_REF', 'storename', 'latitude', 'longitude', 'country', 'city']
SSDB_STRING_COLUMNS = ['SSDB_REF', 'storename','country','city']
SSDB_NUMERIC_COLS = ['latitude', 'longitude']

OTR_ROWS_TO_SKIP = 4

def clean_string_columns_in_df(df, columns):
    """Cleans selected string columns in the input df
    Strips leading/trailing whitespace
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            # Convert to string dtype first
            df[col] = df[col].astype("string")
            # Strip leading/trailing whitespace (handles NaN gracefully)
            df[col] = df[col].str.strip()
        else:
            raise KeyError(f'!!!!WARNING {col} not found in df columns')
    
    return df

def safe_round_to_thousands(series, default=0):
    """Safely round values to nearest thousand, handling NaN and infinite values"""
    try:
        # Convert to numeric, handling errors
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Replace infinite values
        numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with default value
        numeric_series = numeric_series.fillna(default)
        
        # Round to nearest thousand
        return (numeric_series / 1000).round().astype(int) * 1000
    except Exception as e:
        st.warning(f"Rounding failed: {str(e)}")
        return pd.to_numeric(series, errors='coerce').fillna(default)

def create_ss_rent_html(row):
    """Create HTML for SS Rent popup"""
    # Format values with proper NaN handling
    ss_cla = f"{row.get('SS_CLA'):,.0f}" if pd.notna(row.get('SS_CLA')) else 'N/A'
    occ_pct = f"{row.get('Occ % CLA')*100:,.1f}%" if pd.notna(row.get('Occ % CLA')) else 'N/A'
    current_rent = f"{row.get('Current Rent'):,.0f}" if pd.notna(row.get('Current Rent')) else 'N/A'
    anc_inc = f"{row.get('Anc Inc')*100:,.1f}%" if pd.notna(row.get('Anc Inc')) else 'N/A'
    retail = f"{row.get('Retail')*100:,.1f}%" if pd.notna(row.get('Retail')) else 'N/A'
    other = f"{row.get('Other')*100:,.1f}%" if pd.notna(row.get('Other')) else 'N/A'
    insurance = f"{row.get('Insurance')*100:,.1f}%" if pd.notna(row.get('Insurance')) else 'N/A'
    
    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 10px;">
        <h4 style="margin-bottom: 10px;">{row.get('Asset Name', 'Unknown Store')}</h4>
        <p><strong>SS CLA:</strong> {ss_cla}</p>
        <p><strong>Occ % CLA:</strong> {occ_pct}</p>
        <p><strong>Current Rent:</strong> {current_rent}</p>
        <p><strong>Anc Inc:</strong> {anc_inc}</p>
        <p><strong>Retail:</strong> {retail}</p>
        <p><strong>Other:</strong> {other}</p>
        <p><strong>Insurance:</strong> {insurance}</p>
    </div>
    """
    return html

def create_direct_costs_html(row):
    """Create HTML for Direct Costs popup"""
    # Format values with proper NaN handling
    staff = f"{row.get('Staff'):,.0f}" if pd.notna(row.get('Staff')) else 'N/A'
    marketing = f"{row.get('Marketing'):,.0f}" if pd.notna(row.get('Marketing')) else 'N/A'
    utilities = f"{row.get('Utilities'):,.0f}" if pd.notna(row.get('Utilities')) else 'N/A'
    rates = f"{row.get('Rates'):,.0f}" if pd.notna(row.get('Rates')) else 'N/A'
    rent = f"{row.get('Rent'):,.0f}" if pd.notna(row.get('Rent')) else 'N/A'
    
    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 10px;">
        <h4 style="margin-bottom: 10px;">{row.get('Asset Name', 'Unknown Store')}</h4>
        <p><strong>Staff:</strong> {staff}</p>
        <p><strong>Marketing:</strong> {marketing}</p>
        <p><strong>Utilities:</strong> {utilities}</p>
        <p><strong>Rates:</strong> {rates}</p>
        <p><strong>Rent:</strong> {rent}</p>
    </div>
    """
    return html

def read_OTR_file(raw_OTR_uploaded_file):
    """Read and process OTR file into a cleaned Arrow-safe DataFrame"""
    try:
        # 1. Load Excel
        df_raw = pd.read_excel(raw_OTR_uploaded_file, skiprows=OTR_ROWS_TO_SKIP, sheet_name=0)

        if df_raw is None or df_raw.empty:
            st.error("OTR file is empty or unreadable")
            return None

        # 2. Check required columns
        missing_cols = [col for col in REQUIRED_OTR_COLUMNS if col not in df_raw.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in OTR: {missing_cols}")
            return None

        # 3. Clean string columns + ensure Currency unit is all upper
        df_raw = clean_string_columns_in_df(df_raw, OTR_STRING_COLUMNS)
        if "Currency Unit" in df_raw.columns:
            df_raw["Currency Unit"] = df_raw["Currency Unit"].str.upper()

        # 4. Extract year from Valuation date
        if "Valuation date" in df_raw.columns:
            df_raw["Year"] = pd.to_datetime(df_raw["Valuation date"], errors="coerce").dt.year

        # 5. Clean numeric columns
        all_numeric_columns = OTR_ROUNDED_COLUMNS + OTR_PERCENTAGE_COLUMNS + OTR_NUMERIC_COLUMNS + OTR_FLOAT_COLUMNS + ['Year']
        for col in all_numeric_columns:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")    

        # 6. Round selected numeric columns to thousands
        for col in OTR_ROUNDED_COLUMNS:
            if col in df_raw.columns:
                df_raw[col] = safe_round_to_thousands(df_raw[col])  # Fixed: was df[col]

        # 7. Create HTML columns (safe formatting)
        df_raw["html_ss_rent"] = df_raw.apply(create_ss_rent_html, axis=1)
        df_raw["html_direct_costs"] = df_raw.apply(create_direct_costs_html, axis=1)

        # 8. Normalize dtypes (Arrow safe)
        df_raw = df_raw.convert_dtypes()

        st.success(f"✅ Successfully loaded OTR with {df_raw.shape[0]} rows")  # Fixed: was df.shape[0]
        return df_raw  # Fixed: was return df

    except Exception as e:
        st.error(f"Error loading OTR data: {str(e)}")
        return None

def read_ssdb_file(raw_SSDB_uploaded_file):
    """Read and process SSDB file"""
    try:
        df_raw = pd.read_excel(raw_SSDB_uploaded_file, sheet_name=0)
        
        if df_raw is not None:
            # Check required columns
            missing_cols = [col for col in SSDB_REQUIRED_COLUMNS if col not in df_raw.columns]
            if missing_cols:
                st.error(f'Missing columns in SSDB: {missing_cols}')
                return None

            df_raw = clean_string_columns_in_df(df_raw, SSDB_STRING_COLUMNS)

            # Clean numeric columns
            for col in SSDB_NUMERIC_COLS:
                if col in df_raw.columns:
                    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce") 
                else:
                    st.error(f'Missing column in SSDB: {col}')
                    return None

            df = df_raw[SSDB_REQUIRED_COLUMNS].copy()
            SSDB_row_count = df.shape[0]
            st.success(f"✅ Successfully loaded SSDB with {SSDB_row_count} data points")
            return df

    except Exception as e:
        st.error(f'Error loading SSDB data: {str(e)}')
        return None

def create_summary_df(df, display_type):
    """Create summary dataframe with comprehensive statistics"""
    if display_type == 'SS Rent':
        cols = ['SS_CLA', 'Occ % CLA', 'Current Rent', 'Anc Inc', 'Retail', 'Other', 'Insurance']
        # Identify which columns should use percentage formatting
        pct_cols = ['Occ % CLA', 'Anc Inc', 'Retail', 'Other', 'Insurance']
    else:
        cols = ['Staff', 'Marketing', 'Utilities', 'Rates', 'Rent']
        pct_cols = []  # No percentage columns in Direct Costs
    
    summary_data = {}
    for col in cols:
        if col in df.columns:
            # Calculate all required statistics
            min_val = df[col].min()
            q1 = df[col].quantile(0.25)
            median = df[col].quantile(0.5)
            q3 = df[col].quantile(0.75)
            max_val = df[col].max()
            mean_val = df[col].mean()
            
            # Apply appropriate formatting
            if col in pct_cols:
                # Percentage formatting (multiply by 100 and add % sign)
                summary_data[col] = [
                    f"{max_val*100:,.1f}%" if pd.notna(max_val) else "N/A",
                    f"{q3*100:,.1f}%" if pd.notna(q3) else "N/A",
                    f"{median*100:,.1f}%" if pd.notna(median) else "N/A", 
                    f"{q1*100:,.1f}%" if pd.notna(q1) else "N/A",
                    f"{min_val*100:,.1f}%" if pd.notna(min_val) else "N/A",
                    f"{mean_val*100:,.1f}%" if pd.notna(mean_val) else "N/A"
                ]
            else:
                # Standard numeric formatting with thousands separators
                summary_data[col] = [
                    f"{max_val:,.0f}" if pd.notna(max_val) else "N/A",
                    f"{q3:,.0f}" if pd.notna(q3) else "N/A",
                    f"{median:,.0f}" if pd.notna(median) else "N/A",
                    f"{q1:,.0f}" if pd.notna(q1) else "N/A",
                    f"{min_val:,.0f}" if pd.notna(min_val) else "N/A",
                    f"{mean_val:,.0f}" if pd.notna(mean_val) else "N/A"
                ]
    
    # Create summary dataframe
    summary_df = pd.DataFrame(
        summary_data, 
        index=['Max', '75%', 'Median (50%)', '25%', 'Min', 'Average']
    )
        
    return summary_df

# Initialize session state
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_display' not in st.session_state:
    st.session_state.df_display = None
if 'map_center' not in st.session_state:
    st.session_state.map_center = [51.5074, -0.1278]  # Default: London
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 6
if 'display_type' not in st.session_state:
    st.session_state.display_type = 'SS Rent'
if 'size_column' not in st.session_state:
    st.session_state.size_column = 'Current Rent'
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Title
st.title("OTR Application")

# File uploaders
if st.session_state.df_raw is None:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_OTR_file = st.file_uploader(
            "Upload OTR file", 
            type=['xls', 'xlsx', 'xlsm'],
            key=f"otr_uploader_{st.session_state.uploader_key}"
        )
    with col2:
        uploaded_SSDB_file = st.file_uploader(
            "Upload SSDB", 
            type=['xls', 'xlsx', 'xlsm'],
            key=f"ssdb_uploader_{st.session_state.uploader_key}"
        )
    
    if uploaded_OTR_file and uploaded_SSDB_file:
        df_OTR = read_OTR_file(uploaded_OTR_file)
        df_SSDB = read_ssdb_file(uploaded_SSDB_file)
        
        if df_OTR is not None and df_SSDB is not None:
            # Join dataframes
            df_joined = pd.merge(df_SSDB, df_OTR, on='SSDB_REF', how='inner')
            st.session_state.df_raw = df_joined
            st.session_state.df_display = df_joined.copy()
            st.rerun()

# Main application
if st.session_state.df_raw is not None:
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        with st.expander("📅 Year Filter", expanded=True):
            years = sorted(st.session_state.df_raw['Year'].dropna().unique())
            all_years = st.checkbox("All years", value=True)
            
            if not all_years:
                # Use range slider instead of multiselect
                if len(years) > 0:
                    min_year = int(min(years))
                    max_year = int(max(years))
                    
                    selected_year_range = st.slider(
                        "Select year range",
                        min_value=min_year,
                        max_value=max_year,
                        value=(min_year, max_year)
                    )
                    
                    # Convert slider range to list of years for filtering
                    selected_years = [year for year in years if selected_year_range[0] <= year <= selected_year_range[1]]
                else:
                    st.warning("No year data available")
                    selected_years = []
        
        with st.expander("💰 Currency Filter", expanded=True):
            # Get unique values, convert to string, and filter out any 'nan' strings
            raw_currencies = st.session_state.df_raw['Currency Unit'].dropna().unique()
            currencies = [str(x) for x in raw_currencies if str(x) != 'nan']
            
            # Handle case where there are no currencies
            if len(currencies) == 0:
                st.warning("No currency data available")
                selected_currency = None
            else:
                # Sort for consistent ordering
                currencies = sorted(currencies)
                
                # Find default currency index
                try:
                    default_index = currencies.index('GBP')
                except ValueError:
                    default_index = 0
                
                selected_currency = st.radio(
                    "Currency", 
                    currencies, 
                    index=default_index
                )
        
        with st.expander("🌍 Country Filter", expanded=True):
            countries = st.session_state.df_raw['country'].unique()
            selected_countries = st.multiselect("Select countries", countries, default=countries)
        
        with st.expander("📊 Data Display", expanded=True):
            display_type = st.radio("Display Type", ['SS Rent', 'Direct Costs'], index=0)
            st.session_state.display_type = display_type
            
            if display_type == 'SS Rent':
                st.write('Marker size based on SS Rent')
                st.session_state.size_column = 'Current Rent'  # Fixed: Set the size column for SS Rent
            else:
                st.session_state.size_column = st.radio("Marker Size Based On", 
                                                       ['Staff', 'Marketing', 'Utilities', 'Rates', 'Rent'],
                                                       index=4)
            
            # Size and occupancy filters
            occ_min = float(st.session_state.df_raw['Occ % CLA'].min()) * 100
            occ_max = float(st.session_state.df_raw['Occ % CLA'].max()) * 100
            occ_filter = st.slider("Occ % CLA Filter", occ_min, occ_max, (occ_min, occ_max))
            
            ss_cla_min = float(st.session_state.df_raw['SS_CLA'].min())
            ss_cla_max = float(st.session_state.df_raw['SS_CLA'].max())
            ss_cla_filter = st.slider("SS CLA Filter", ss_cla_min, ss_cla_max, (ss_cla_min, ss_cla_max))
    
    # Apply filters
    df_filtered = st.session_state.df_raw.copy()
    
    # Year filter
    if not all_years:
        df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]
    
    # Currency filter - Fixed: Check Area unit instead of Currency Unit
    if selected_currency:
        df_filtered = df_filtered[df_filtered['Currency Unit'] == selected_currency]
    
    # Country filter
    df_filtered = df_filtered[df_filtered['country'].isin(selected_countries)]
    
    # Occupancy and SS CLA filters
    df_filtered = df_filtered[
        (df_filtered['Occ % CLA'] * 100 >= occ_filter[0]) & 
        (df_filtered['Occ % CLA'] * 100 <= occ_filter[1]) &
        (df_filtered['SS_CLA'] >= ss_cla_filter[0]) & 
        (df_filtered['SS_CLA'] <= ss_cla_filter[1])
    ]
    
    st.session_state.df_display = df_filtered
    
    # Main content tabs
    tab1, tab2 = st.tabs(["Map", "Data"])
    
    with tab1:
        if len(df_filtered) == 0:
            st.warning("No data to display with current filters")
        else:
            # Create map
            center_list = st.session_state.map_center
            if isinstance(center_list, dict):
                center_list = [center_list['lat'], center_list['lng']]
            
            m = folium.Map(location=center_list, zoom_start=st.session_state.map_zoom)
            feature_group = folium.FeatureGroup(name="OTR Data")
            
            # Add markers
            for idx, row in st.session_state.df_display.iterrows():
                if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                    if display_type == 'SS Rent':
                        html_content = row['html_ss_rent']
                        size_col = 'Current Rent'
                        color = 'blue'
                    else:
                        html_content = row['html_direct_costs']
                        size_col = st.session_state.size_column
                        color = 'green'
                    
                    # Calculate radius with better error handling
                    try:
                        size_value = row[size_col]
                        max_value = st.session_state.df_display[size_col].max()
                        if pd.notna(size_value) and pd.notna(max_value) and max_value > 0:
                            radius = 5 + (size_value / max_value * 15)
                        else:
                            radius = 5
                    except (KeyError, ZeroDivisionError):
                        radius = 5
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        popup=folium.Popup(html_content, max_width=300)
                    ).add_to(feature_group)
            
            feature_group.add_to(m)
            folium.LayerControl().add_to(m)
            
            # Display map
            map_data = st_folium(m, width=1200, height=600, returned_objects=[])
            
            # Update session state from map interactions
            if map_data:
                if "center" in map_data:
                    st.session_state.map_center = [map_data["center"]["lat"], map_data["center"]["lng"]]
                if "zoom" in map_data:
                    st.session_state.map_zoom = map_data["zoom"]
    
    with tab2:
        if len(df_filtered) == 0:
            st.warning("No data to display with current filters")
        else:
            # Data tab
            st.header("Data Summary")
            summary_df = create_summary_df(st.session_state.df_display, display_type)
            st.dataframe(summary_df, use_container_width=True)
            
            st.header("Detailed Data")
            if display_type == 'SS Rent':
                display_cols = ['Asset Name', 'Year', 'SS_CLA', 'Occ % CLA', 'Current Rent', 
                               'Anc Inc', 'Retail', 'Other', 'Insurance']
            else:
                display_cols = ['Asset Name', 'Year', 'Staff', 'Marketing', 'Utilities', 'Rates', 'Rent']
            
            detailed_df = st.session_state.df_display[display_cols].copy()
            
            # Format percentage columns for display
            if display_type == 'SS Rent':
                for col in ['Occ % CLA', 'Anc Inc', 'Retail', 'Other', 'Insurance']:
                    if col in detailed_df.columns:
                        detailed_df[col] = detailed_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
            
            st.dataframe(detailed_df, use_container_width=True)

# Reset button
if st.sidebar.button("🔄 Clear input data"):
    # Clear all session state except uploader_key
    uploader_key = st.session_state.get('uploader_key', 0)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Increment the uploader key to force new file uploader widgets
    st.session_state.uploader_key = uploader_key + 1
    # Reinitialize essential session state
    st.session_state.df_raw = None
    st.session_state.df_display = None
    st.session_state.map_center = [51.5074, -0.1278]
    st.session_state.map_zoom = 6
    st.session_state.display_type = 'SS Rent'
    st.session_state.size_column = 'Current Rent'
    st.rerun()