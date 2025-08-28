import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from folium.plugins import FeatureGroupSubGroup
import re

###########################################

st.set_page_config(page_title="OTR App", 
                   layout="wide",
                   page_icon="🔓",
                   initial_sidebar_state="collapsed")


###########################################

# Custom CSS to reduce gap between top of page and title
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

##################################################

# Constants #

SQ_FT_TO_SQ_M = 10.764
ROUNDING_RENT_SQM = 5
ROUNDING_RENT_SQFT = 0.5
ROUNDING_AREA_SQM = 1000
ROUNDING_AREA_SQFT = 100

AREA_UNIT_COLUMN = 'Area unit'
SS_RENT_COLUMN_CLEANED = 'SS_Current Rent'

REQUIRED_OTR_COLUMNS = ['Asset Name', 'SSDB_REF', AREA_UNIT_COLUMN,'Currency Unit', 'Valuation date', 'MLA', 'CLA',
                        'SS_CLA', 'SS_MLA', 'SS_Occ area', 'SS_Occ % CLA', SS_RENT_COLUMN_CLEANED,
                        'Anc Inc','Retail', 'Other', 'Insurance',
                        'Staff', 'Marketing', 'Utilities', 'Rates', 'Rent',
                        'SS Unit Count', 'Avg SS Unit Size', 'SS Revenue', 'Total Rev',
                        'Opex Total', 'EBITDAR', 'EBITDA']

OTR_STRING_COLUMNS = ['Asset Name', 'SSDB_REF', 'Currency Unit']
OTR_ROUNDED_COLUMNS = ['Staff', 'Marketing', 'Utilities', 'Rates', 'Rent']
OTR_PERCENTAGE_COLUMNS = ['Anc Inc', 'Retail', 'Other', 'Insurance','SS_Occ % CLA']
OTR_NUMERIC_COLUMNS = ['Anc Inc', 'Retail', 'Other', 'Insurance', 'SS_CLA']
OTR_FLOAT_COLUMNS = ['SS_Current Rent']
# These will be converted to _sqm and _sqft depending on Area Unit
OTR_AREA_COLS = ['MLA', 'CLA',  'SS_CLA', 'SS_MLA', 'SS_Occ area',  'SS_Current Rent', 'Avg SS Unit Size']
OTR_RENT_COLS = ['SS_Current Rent'] # only one item but do this so can be scaled later
OTR_ROWS_TO_SKIP = 4

SSDB_REQUIRED_COLUMNS = ['SSDB_REF', 'storename', 'latitude', 'longitude', 'country', 'city']
SSDB_STRING_COLUMNS = ['SSDB_REF', 'storename','country','city']
SSDB_NUMERIC_COLS = ['latitude', 'longitude']

DF_JOINED_OUTPUT_COLUMNS = ['storename', 'latitude', 'longitude', 'country', 'city','Year',
                            'SS_CLA_sqm', 'SS_CLA_sqft', 'SS_Occ % CLA',  
                            'SS_Current Rent_sqm', 'SS_Current Rent_sqft',
                            'Avg SS Unit Size_sqm', 'Avg SS Unit Size_sqft',
                          'Staff','Marketing', 'Utilities', 'Rates', 'Rent',
                           'html_ss_rent', 'html_direct_costs']

DF_FILTERED_COLUMNS_SQM = ['storename', 'latitude', 'longitude', 'country', 'city','Year',
                            'SS_CLA_sqm', 'SS_Occ % CLA',  'Currency Unit',
                            'SS_Current Rent_sqm', 
                            'Avg SS Unit Size_sqm', 
                             'Anc Inc', 'Retail', 'Other', 'Insurance',
                          'Staff','Marketing', 'Utilities', 'Rates', 'Rent',
                           'html_ss_rent', 'html_direct_costs']

DF_FILTERED_COLUMNS_SQFT = ['storename', 'latitude', 'longitude', 'country', 'city','Year',
                            'SS_CLA_sqft', 'SS_Occ % CLA',  'Currency Unit',
                             'SS_Current Rent_sqft',
                             'Avg SS Unit Size_sqft',
                             'Anc Inc', 'Retail', 'Other', 'Insurance',
                          'Staff','Marketing', 'Utilities', 'Rates', 'Rent',
                           'html_ss_rent', 'html_direct_costs']

RENAME_COLUMNS_DICT = {
    "SS_CLA_sqm": "SS_CLA",
    "SS_CLA_sqft": "SS_CLA",
    "SS_Current Rent_sqm": "SS_Current Rent",
    "SS_Current Rent_sqft": "SS_Current Rent",
    "Avg SS Unit Size_sqm": "Avg SS Unit Size",
    "Avg SS Unit Size_sqft": "Avg SS Unit Size",
    "storename": "Store name",
}

# Note this is after the renaming via RENAME_COLUMNS_DICT
DISPLAY_COLUMNS_SS_RENT = ['Store name', 'Year', 'SS_CLA', 'SS_Occ % CLA', 'SS_Current Rent', 
                               'Anc Inc', 'Retail', 'Other', 'Insurance']
DISPLAY_COLUMNS_DIRECT_COSTS = ['Store name', 'Year', 'Staff', 'Marketing', 'Utilities', 'Rates', 'Rent']

TILE_LAYER = 'CartoDB Positron'

HTML_BODY_FONT_SIZE = 8
HTML_H4_FONT_SIZE = 10
HTML_LINE_HEIGHT = 1.0 # This controls vertical space between lines smaller = tighter

MIN_CIRCLE_RADIUS = 2

MAP_ZOOM_DEFAULT = 6
MAP_CENTER_DEFAULT = [51.5074, -0.1278]

#######################################################

# Functions #

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


def convert_areas_based_on_area_type(df, area_cols, area_unit_col=AREA_UNIT_COLUMN):
    """
    Convert area columns to both sqm and sqft based on area unit type.
    
    Args:
        df: pandas DataFrame
        area_cols: list of area column names to process
        area_unit_col: column name containing area unit information
    
    Returns:
        pandas DataFrame with converted columns
    """
    for col in area_cols:
        # Create sqm column
        sqm_col = f"{col}_sqm"
        df[sqm_col] = np.where(
            df[area_unit_col] == 'sq m',
            df[col],
            np.where(
                df[area_unit_col] == 'sq ft',
                df[col] / SQ_FT_TO_SQ_M,
                pd.NA  # Arrow-compliant null
            )
        )
        # Round sqm values
        df[sqm_col] = (df[sqm_col] / ROUNDING_AREA_SQM).round() * ROUNDING_AREA_SQM
        
        # Create sqft column
        sqft_col = f"{col}_sqft"
        df[sqft_col] = df[sqm_col] * SQ_FT_TO_SQ_M
        # Round sqft values
        df[sqft_col] = (df[sqft_col] / ROUNDING_AREA_SQFT).round() * ROUNDING_AREA_SQFT
    
    return df

def convert_rents_based_on_area_type(df, rent_cols, area_unit_col=AREA_UNIT_COLUMN):
    """
    Convert rent columns to both per sqm and per sqft based on area unit type.
    
    Args:
        df: pandas DataFrame
        rent_cols: list of rent column names to process
        area_unit_col: column name containing area unit information
    
    Returns:
        pandas DataFrame with converted columns
    """
    for col in rent_cols:
        # Create rent per sqm column
        rent_sqm_col = f"{col}_sqm"
        df[rent_sqm_col] = np.where(
            df[area_unit_col] == 'sq m',
            df[col],
            np.where(
                df[area_unit_col] == 'sq ft',
                df[col] * SQ_FT_TO_SQ_M,  # Convert from per sqft to per sqm
                pd.NA  # Arrow-compliant null
            )
        )
        # Round rent per sqm values
        df[rent_sqm_col] = (df[rent_sqm_col] / ROUNDING_RENT_SQM).round() * ROUNDING_RENT_SQM
        
        # Create rent per sqft column
        rent_sqft_col = f"{col}_sqft"
        df[rent_sqft_col] = df[rent_sqm_col] / SQ_FT_TO_SQ_M
        # Round rent per sqft values
        df[rent_sqft_col] = (df[rent_sqft_col] / ROUNDING_RENT_SQFT).round() * ROUNDING_RENT_SQFT
    
    return df


def create_ss_rent_html(row):
    """Create HTML for SS Rent popup"""
    # Format values with proper NaN handling
    storename = row.get('storename', 'Unknown Store')
    ss_cla = f"{row.get('SS_CLA'):,.0f}" if pd.notna(row.get('SS_CLA')) else 'N/A'
    occ_pct = f"{row.get('SS_Occ % CLA')*100:,.1f}%" if pd.notna(row.get('SS_Occ % CLA')) else 'N/A'
    current_rent = f"{row.get('SS_Current Rent'):,.0f}" if pd.notna(row.get('SS_Current Rent')) else 'N/A'
    anc_inc = f"{row.get('Anc Inc')*100:,.1f}%" if pd.notna(row.get('Anc Inc')) else 'N/A'
    retail = f"{row.get('Retail')*100:,.1f}%" if pd.notna(row.get('Retail')) else 'N/A'
    other = f"{row.get('Other')*100:,.1f}%" if pd.notna(row.get('Other')) else 'N/A'
    insurance = f"{row.get('Insurance')*100:,.1f}%" if pd.notna(row.get('Insurance')) else 'N/A'
    area_unit = row.get(AREA_UNIT_COLUMN, '')
    
    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 10px; line-height: {HTML_LINE_HEIGHT}; font-size: {HTML_BODY_FONT_SIZE}px">
        <h4 style="margin-bottom: 10px; font-size: {HTML_H4_FONT_SIZE}px"><strong>{storename}</strong></h4>
        <p><strong>SS CLA:</strong> {ss_cla} {area_unit}</p>
        <p><strong>Occ % CLA:</strong> {occ_pct}</p>
        <p><strong>Current Rent:</strong> {current_rent} per {area_unit}</p>
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
    storename = row.get('storename', 'Unknown Store')
    staff = f"{row.get('Staff'):,.0f}" if pd.notna(row.get('Staff')) else 'N/A'
    marketing = f"{row.get('Marketing'):,.0f}" if pd.notna(row.get('Marketing')) else 'N/A'
    utilities = f"{row.get('Utilities'):,.0f}" if pd.notna(row.get('Utilities')) else 'N/A'
    rates = f"{row.get('Rates'):,.0f}" if pd.notna(row.get('Rates')) else 'N/A'
    rent = f"{row.get('Rent'):,.0f}" if pd.notna(row.get('Rent')) else 'N/A'
    
    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 10px; line-height: {HTML_LINE_HEIGHT}; font-size: {HTML_BODY_FONT_SIZE}px">
        <h4 style="margin-bottom: 10px; font-size: {HTML_H4_FONT_SIZE}px"><strong>{storename}</strong></h4>
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
        # Read in the Excel file
        df_raw = pd.read_excel(raw_OTR_uploaded_file, skiprows=OTR_ROWS_TO_SKIP, sheet_name=0)

        if df_raw is None or df_raw.empty:
            st.error("OTR file is empty or unreadable")
            return None

        # Check required columns
        missing_cols = [col for col in REQUIRED_OTR_COLUMNS if col not in df_raw.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in OTR: {missing_cols}")
            return None

        # Clean string columns + ensure Currency unit is all UC and all Area Units LC
        df_raw = clean_string_columns_in_df(df_raw, OTR_STRING_COLUMNS)
        if "Currency Unit" in df_raw.columns:
            df_raw["Currency Unit"] = df_raw["Currency Unit"].str.upper()
        if AREA_UNIT_COLUMN in df_raw.columns:
            df_raw[AREA_UNIT_COLUMN] = df_raw[AREA_UNIT_COLUMN].str.lower()

        # Extract year from Valuation date
        if "Valuation date" in df_raw.columns:
            df_raw["Year"] = pd.to_datetime(df_raw["Valuation date"], errors="coerce").dt.year

        # Clean numeric columns
        all_numeric_columns = OTR_ROUNDED_COLUMNS + OTR_PERCENTAGE_COLUMNS + OTR_NUMERIC_COLUMNS + OTR_FLOAT_COLUMNS + ['Year']
        for col in all_numeric_columns:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")    

        # Round selected numeric columns to thousands
        for col in OTR_ROUNDED_COLUMNS:
            if col in df_raw.columns:
                df_raw[col] = safe_round_to_thousands(df_raw[col])  # Fixed: was df[col]

        # with the data cleaned now create rents and areas in _sqm and sqft
        df_raw =  convert_areas_based_on_area_type(df_raw, OTR_AREA_COLS, area_unit_col=AREA_UNIT_COLUMN)
        df_raw = convert_rents_based_on_area_type(df_raw, OTR_RENT_COLS, area_unit_col=AREA_UNIT_COLUMN)

        # Normalize dtypes (Arrow safe)
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
        cols = ['SS_CLA', 'Occ % CLA', SS_RENT_COLUMN_CLEANED, 'Anc Inc', 'Retail', 'Other', 'Insurance']
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


#######################################################

# Session State set up

# Initialize session states
SESSION_STATE_KEYS_SET_TO_NONE = ['df_OTR', 'df_OTR_sqm', 'df_OTR_sqft', 'df_OTR_Selected_Area', 
                                  'df_display', 'df_raw', 'df_display']
for key in SESSION_STATE_KEYS_SET_TO_NONE:
    if key not in st.session_state:
        st.session_state[key] = None

if 'map_center' not in st.session_state:
    st.session_state.map_center = [51.5074, -0.1278]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 6
if 'display_type' not in st.session_state:
    st.session_state.display_type = 'SS Rent'
if 'size_column' not in st.session_state:
    st.session_state.size_column = SS_RENT_COLUMN_CLEANED
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'all_years' not in st.session_state:
    st.session_state.all_years = True
if 'selected_year_range' not in st.session_state:
    st.session_state.selected_year_range = (2010, 2023)
if 'selected_currency' not in st.session_state:
    st.session_state.selected_currency = 'GBP'
if 'selected_countries' not in st.session_state:
    st.session_state.selected_countries = []

# No Title to make sure map is higher up page
# st.title("OTR Application")

# File uploaders
if st.session_state.df_OTR_Selected_Area is None:
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

            # Add HTML
            df_joined["html_ss_rent"] = df_joined.apply(create_ss_rent_html, axis=1)
            df_joined["html_direct_costs"] = df_joined.apply(create_direct_costs_html, axis=1)

            # Create sqm and sqft versions
            df_OTR_sqm = df_joined[DF_FILTERED_COLUMNS_SQM].copy().rename(columns=RENAME_COLUMNS_DICT)
            df_OTR_sqft = df_joined[DF_FILTERED_COLUMNS_SQFT].copy().rename(columns=RENAME_COLUMNS_DICT)

            # Store in session
            st.session_state.df_OTR_sqm = df_OTR_sqm
            st.session_state.df_OTR_sqft = df_OTR_sqft

            # Default: sq ft
            st.session_state.df_OTR_Selected_Area = df_OTR_sqft.copy()

            st.rerun()

# Main application
if st.session_state.df_OTR_Selected_Area is not None:
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")

        # Area unit selector (default = sq ft)
        area_unit = st.radio("Select Area Unit", ["sq m", "sq ft"], index=1)

        if area_unit == "sq m":
            st.session_state.df_OTR_Selected_Area = st.session_state.df_OTR_sqm.copy()
        else:
            st.session_state.df_OTR_Selected_Area = st.session_state.df_OTR_sqft.copy()
        
        with st.expander("📅 Year Filter", expanded=True):
            years = sorted(st.session_state.df_OTR_Selected_Area['Year'].dropna().unique())
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
            raw_currencies = st.session_state.df_OTR_Selected_Area['Currency Unit'].dropna().unique()
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
            countries = st.session_state.df_OTR_Selected_Area['country'].unique()
            selected_countries = st.multiselect("Select countries", countries, default=countries)
        
        with st.expander("📊 Data Display", expanded=True):
            display_type = st.radio("Display Type", ['SS Rent', 'Direct Costs'], index=0)
            st.session_state.display_type = display_type
            
            if display_type == 'SS Rent':
                st.write('Marker size based on SS Rent')
                st.session_state.size_column = SS_RENT_COLUMN_CLEANED # Fixed: Set the size column for SS Rent
            else:
                st.session_state.size_column = st.radio("Marker Size Based On", 
                                                       ['Staff', 'Marketing', 'Utilities', 'Rates', 'Rent'],
                                                       index=4)
            
            # Size and occupancy filters
            occ_min = float(st.session_state.df_OTR_Selected_Area['SS_Occ % CLA'].min()) * 100
            occ_max = float(st.session_state.df_OTR_Selected_Area['SS_Occ % CLA'].max()) * 100
            occ_filter = st.slider("Occ % CLA Filter", occ_min, occ_max, (occ_min, occ_max))
            
            ss_cla_min = float(st.session_state.df_OTR_Selected_Area['SS_CLA'].min())
            ss_cla_max = float(st.session_state.df_OTR_Selected_Area['SS_CLA'].max())
            ss_cla_filter = st.slider("SS CLA Filter", ss_cla_min, ss_cla_max, (ss_cla_min, ss_cla_max))
    
    # Apply filters
    df_filtered = st.session_state.df_OTR_Selected_Area.copy()
    
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
        (df_filtered['SS_Occ % CLA'] * 100 >= occ_filter[0]) & 
        (df_filtered['SS_Occ % CLA'] * 100 <= occ_filter[1]) &
        (df_filtered['SS_CLA'] >= ss_cla_filter[0]) & 
        (df_filtered['SS_CLA'] <= ss_cla_filter[1])
    ]
    
    # Once we have completed the updated filtering on the df_filtered use this to adjust the map
    if len(df_filtered) > 0:
        mean_lat = df_filtered['latitude'].mean()
        mean_lon = df_filtered['longitude'].mean()
        st.session_state.map_center = [mean_lat, mean_lon]
        st.session_state.map_zoom = MAP_ZOOM_DEFAULT 
    else:
        # Fallback to default if no data
        st.session_state.map_center = MAP_ZOOM_DEFAULT
        st.session_state.map_zoom = MAP_CENTER_DEFAULT

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
            
            m = folium.Map(location=st.session_state.map_center, 
                           zoom_start=st.session_state.map_zoom, 
                           tiles=TILE_LAYER)
            feature_group = folium.FeatureGroup(name="OTR Data")
            
            # Add markers
            for idx, row in st.session_state.df_display.iterrows():
                if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                    if display_type == 'SS Rent':
                        html_content = row['html_ss_rent']
                        size_col = SS_RENT_COLUMN_CLEANED
                        color = 'green'
                    else:
                        html_content = row['html_direct_costs']
                        size_col = st.session_state.size_column
                        color = 'red'
                    
                    # Calculate radius with better error handling
                    try:
                        size_value = row[size_col]
                        max_value = st.session_state.df_display[size_col].max()
                        if pd.notna(size_value) and pd.notna(max_value) and max_value > 0:
                            radius = MIN_CIRCLE_RADIUS + (size_value / max_value * 15)
                        else:
                            radius = MIN_CIRCLE_RADIUS
                    except (KeyError, ZeroDivisionError):
                        radius = MIN_CIRCLE_RADIUS
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        popup=folium.Popup(html_content, max_width=500)
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
                display_cols = DISPLAY_COLUMNS_SS_RENT
            else:
                display_cols = DISPLAY_COLUMNS_DIRECT_COSTS
            
            detailed_df = st.session_state.df_display[display_cols].copy()
            
            # Format percentage columns for display
            if display_type == 'SS Rent':
                for col in ['SS_Occ % CLA', 'Anc Inc', 'Retail', 'Other', 'Insurance']:
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
    st.session_state.df_OTR_Selected_Area = None
    st.session_state.map_center = [51.5074, -0.1278]
    st.session_state.map_zoom = 6
    st.session_state.display_type = 'SS Rent'
    st.session_state.size_column = SS_RENT_COLUMN_CLEANED
    st.rerun()
