I think your idea for integrating the two apps is excellent. It creates a seamless user experience where users can search and filter funds at the top, select multiple tickers from the table with visual indicators for the order of selection, and then view detailed performance charts below. Having the charts appear only after selecting tickers makes sense to avoid displaying empty or irrelevant information. I've implemented this by combining the code from both apps, using st.data_editor for the table to allow checkbox selection, and preserving the selection order with session state. Different numbered emojis indicate the monitoring order. The filters are placed above the respective charts in the main area, as the sidebar is collapsed. For the single asset view, you can choose which selected ticker to view, and the head-to-head appears if more than one is selected.

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Integrated High-Yield Terminal", layout="wide", initial_sidebar_state="collapsed")

# --- MERGED CUSTOM CSS (Combining both apps, with terminal overrides where conflicting) ---
st.markdown("""
    <style>
        /* From Search App */
        .stApp {
            background-color: #0D1117;
            color: #E6EDF3;
        }
        label {display: none !important;}
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] {
            background-color: #1E293B !important;
            border-color: #30363d !important;
            color: #E6EDF3 !important;
            border-radius: 6px !important;
            min-height: 45px !important;
        }
        input { color: #E6EDF3 !important; font-weight: bold !important; }
        div[data-baseweb="select"] div { color: #E6EDF3 !important; }
        ul[role="listbox"], div[data-baseweb="menu"] {
            background-color: #1E293B !important;
            border: 1px solid #30363d !important;
        }
        li[role="option"] {
            color: #E6EDF3 !important;
            background-color: #1E293B !important;
        }
        li[role="option"]:hover, li[role="option"][aria-selected="true"] {
            background-color: #8AC7DE !important;
            color: #0D1117 !important;
        }
        .stMultiSelect span[data-baseweb="tag"] {
            background-color: #8AC7DE !important;
            color: #0D1117 !important;
            font-weight: bold;
        }
        /* Yield Slider */
        div[data-testid="stSlider"] {
            background-color: #1E293B;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 5px 10px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        div[data-testid="stSlider"] label {
            display: block !important;
            color: #E6EDF3 !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            margin-bottom: 5px !important;
        }
        div[data-baseweb="slider"] div[style*="background-color: rgb(211, 211, 211)"] {
            background-color: #4B5563 !important;
        }
        div[data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"] {
            background-color: #8AC7DE !important;
        }
        div[role="slider"] {
            background-color: #8AC7DE !important;
            border: 2px solid #E6EDF3 !important;
            box-shadow: 0 0 10px rgba(138, 199, 222, 0.4);
            width: 20px !important;
            height: 20px !important;
        }
        div[data-baseweb="slider"] span {
            color: #E6EDF3 !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }
        /* Table Styling (Merged) */
        div[data-testid="stDataFrameResizable"] {
            background-color: #0D1117 !important;
            border: 1px solid #30363d;
            border-radius: 5px;
        }
        div[data-testid="stDataFrameResizable"] div[role="columnheader"] {
            background-color: #1E293B !important;
            color: #8AC7DE !important;
            font-weight: bold;
            border-bottom: 1px solid #30363d;
        }
        div[data-testid="stDataFrameResizable"] div[role="gridcell"] {
            background-color: #0D1117 !important;
            color: #E6EDF3 !important;
            border-bottom: 1px solid #30363d;
        }
        div[data-testid="stDataFrameResizable"] div[role="row"]:hover div[role="gridcell"] {
            background-color: #161B22 !important;
        }
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 0rem; max-width: 1400px;}
        ::placeholder { color: #94A3B8 !important; opacity: 1; }
        .row-widget.stTextInput { margin-bottom: -10px !important; }
        .stMultiSelect { margin-top: 0 !important; }
        .element-container { margin-top: 0 !important; margin-bottom: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important; }
        .stHorizontalBlock > div { margin-top: 0 !important; }

        /* From Terminal App (Overrides where needed) */
        div[data-testid="stMetric"] {
            background-color: #1E293B;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 8px 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            min-height: 80px;
            transition: transform 0.2s;
        }
        div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetric"] {
            background-color: #1a2e35 !important;
            border: 1px solid #F59E0B !important;
        }
        div[data-testid="column"]:nth-of-type(5) div[data-testid="stMetric"] {
            background-color: #0D1117 !important;
            border: 2px solid #00C805 !important;
            transform: scale(1.15);
            z-index: 10;
            margin-left: 10px;
        }
        div[data-testid="column"]:nth-of-type(5) div[data-testid="stMetricValue"] div {
            font-size: 1.8rem !important;
        }
        div[data-testid="stMetricLabel"] p { color: #8AC7DE !important; font-size: 0.85rem !important; font-weight: 600 !important; }
        div[data-testid="stMetricValue"] div { color: #FFFFFF !important; font-size: 1.5rem !important; font-weight: 700 !important; }
        div[data-testid="stMetricDelta"] svg { transform: scale(1.2); }
        div[data-testid="stMetricDelta"] > div {
            font-size: 1.1rem !important;
            font-weight: 800 !important;
            filter: brightness(1.2);
        }
        [data-testid="stMetricLabel"] svg {
            fill: #E6EDF3 !important;
            opacity: 0.9 !important;
            width: 16px !important;
            height: 16px !important;
        }
        [data-testid="stMetricLabel"]:hover svg {
            fill: #F59E0B !important;
            opacity: 1.0 !important;
        }
        div[role="tooltip"] {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
            border: 1px solid #8AC7DE !important;
            border-radius: 6px !important;
            font-size: 0.9rem !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
        }
        div[role="tooltip"] > div {
            background-color: #1E293B !important;
        }
        div[data-baseweb="calendar"] { background-color: #1E293B !important; color: #FFFFFF !important; border: 1px solid #30363d !important; }
        div[data-baseweb="calendar"] > div { background-color: #1E293B !important; }
        div[data-baseweb="select"] div { color: #FFFFFF !important; }
        div[data-baseweb="calendar"] button svg { fill: #8AC7DE !important; }
        div[data-baseweb="calendar"] button { background-color: transparent !important; }
        div[data-baseweb="calendar"] div[role="grid"] div { color: #E6EDF3 !important; }
        div[data-baseweb="calendar"] button[aria-label] { color: #FFFFFF !important; }
        div[data-baseweb="calendar"] [aria-selected="true"] { background-color: #8AC7DE !important; color: #0D1117 !important; font-weight: bold !important; }
        div[data-baseweb="calendar"] [aria-selected="false"]:hover { background-color: #30363d !important; color: #FFFFFF !important; }
        h1, h2, h3, h4, h5, h6, p, label { color: #E6EDF3 !important; }
        div[data-baseweb="select"] > div, div[data-testid="stDateInput"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
            background-color: #1E293B !important;
            border-color: #30363d !important;
            color: #FFFFFF !important;
            font-weight: bold !important;
            border-radius: 6px !important;
            min-height: 40px !important;
        }
        input { color: #FFFFFF !important; font-weight: bold !important; }
        .stSelectbox svg, .stDateInput svg { fill: #8AC7DE !important; }
        .stSidebar .element-container { margin-top: 0rem !important; margin-bottom: 0.5rem !important; }
        .stSidebar .stSelectbox, .stSidebar .stDateInput, .stSidebar .stRadio, .stSidebar .stNumberInput { padding-top: 0rem !important; padding-bottom: 0rem !important; }
        .stSidebar .stCheckbox label { font-weight: bold; color: #8AC7DE !important; }
        div[data-testid="stDataFrameResizable"] div[data-testid="stVerticalBlock"] { overflow: hidden !important; }
        div[data-testid="stDataFrameResizable"] ::-webkit-scrollbar { display: none !important; }
        @media (min-width: 1200px) {
            .block-container { padding-left: 3rem !important; padding-right: 3rem !important; padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 100% !important; }
        }
        @media (max-width: 1199px) {
            .block-container { padding-top: 4rem !important; padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100vw !important; min-width: 100vw !important; }
            div[data-testid="stMetric"] {
                background-color: transparent !important;
                border: none !important;
                border-bottom: 1px solid #30363d !important;
                border-radius: 0px !important;
                box-shadow: none !important;
                padding: 5px 0px !important;
                min-height: auto !important;
                margin-bottom: 0px !important;
            }
            div[data-testid="stMetric"] > div {
                display: flex !important;
                flex-direction: row !important;
                justify-content: space-between !important;
                align-items: center !important;
                width: 100% !important;
            }
            div[data-testid="stMetricLabel"] {
                order: 1 !important;
            }
            div[data-testid="stMetricLabel"] p {
                margin-bottom: 0px !important;
            }
            div[data-testid="stMetricDelta"] {
                order: 2 !important;
                margin-left: auto !important;
                margin-right: 10px !important;
                margin-bottom: 0px !important;
            }
            div[data-testid="stMetricValue"] {
                order: 3 !important;
                text-align: right !important;
            }
            div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetric"] {
                background-color: transparent !important;
                border: none !important;
                border-bottom: 1px solid #F59E0B !important;
            }
            div[data-testid="column"]:nth-of-type(5) div[data-testid="stMetric"] {
                background-color: transparent !important;
                border: none !important;
                border-bottom: 1px solid #00C805 !important;
                transform: none !important;
                margin-left: 0px !important;
                margin-top: 0px !important;
            }
            div[data-testid="column"]:nth-of-type(5) div[data-testid="stMetricValue"] div {
                font-size: 1.5rem !important;
            }
        }
        #MainMenu {visibility: hidden !important;}
        .viewerBadge_container__1QSob {display: none !important;}
        .viewerBadge_link__1SlnQ {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# Hide Streamlit badges
st.components.v1.html('''
    <script>
    window.top.document.querySelectorAll(`[href*="streamlit.io"]`).forEach(e => e.setAttribute("style", "display: none;"));
    </script>
''', height=0)

# --- HELPER FUNCTIONS (From Terminal App) ---
@st.cache_data(ttl=3600)
def fetch_overlay_data(ticker, start_date, end_date):
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(
            start=start_date,
            end=end_date + timedelta(days=1),
            actions=True,
            auto_adjust=False
        )
        if hist.empty:
            return None, None
        df_u = hist.reset_index()[['Date', 'Close']].rename(columns={'Close': 'Closing Price'})
        df_u['Date'] = pd.to_datetime(df_u['Date']).dt.tz_localize(None)
        df_u['Ticker'] = ticker
        divs = hist[hist['Dividends'] > 0]['Dividends'].reset_index()
        if not divs.empty:
            df_h = divs.rename(columns={'Date': 'Date of Pay', 'Dividends': 'Amount'})
            df_h['Date of Pay'] = pd.to_datetime(df_h['Date of Pay']).dt.tz_localize(None)
            df_h['Ticker'] = ticker
        else:
            df_h = pd.DataFrame(columns=['Date of Pay', 'Amount', 'Ticker'])
        return df_u, df_h
    except Exception:
        return None, None

@st.cache_data(ttl=3600)
def load_base_sheets():
    try:
        h_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTKjBzr7QuJk9g7TR6p0-_GdPQDvesG9a1KTny6y5IyK0Z-G0_C98T-AfUyaAdyDB11h3vdpgc_h3Hh/pub?gid=1266509012&single=true&output=csv"
        m_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTKjBzr7QuJk9g7TR6p0-_GdPQDvesG9a1KTny6y5IyK0Z-G0_C98T-AfUyaAdyDB11h3vdpgc_h3Hh/pub?gid=618318322&single=true&output=csv"
        df_m = pd.read_csv(m_url)
        df_h_sheet = pd.read_csv(h_url)
        m_rename_map = {'Fund Strategy': 'Strategy', 'Asset Class': 'Strategy', 'Fund Name': 'Company', 'Name': 'Company'}
        df_m = df_m.rename(columns=m_rename_map)
        if 'Ticker' not in df_m.columns:
            return None, None
        df_m['Ticker'] = df_m['Ticker'].astype(str).str.strip().str.upper()
        df_m = df_m[df_m['Ticker'].str.match(r'^[A-Z0-9\-]{1,8}$', na=False)]
        bad_headers = ['DEFIANCE', 'YIELDMAX', 'ROUNDHILL', 'KURV', 'PROSHARES', 'GLOBALX', 'REX', 'CALAMOS', 'INVESCO', 'SIMPLIFY', 'VANECK']
        df_m = df_m[~df_m['Ticker'].isin(bad_headers)]
        h_rename_map = {'Pay Date': 'Date of Pay', 'Payment Date': 'Date of Pay', 'Payout Date': 'Date of Pay', 'Date': 'Date of Pay'}
        df_h_sheet = df_h_sheet.rename(columns=h_rename_map)
        if 'Date of Pay' in df_h_sheet.columns and 'Ticker' in df_h_sheet.columns:
            df_h_sheet['Date of Pay'] = pd.to_datetime(df_h_sheet['Date of Pay']).dt.tz_localize(None)
            df_h_sheet['Ticker'] = df_h_sheet['Ticker'].astype(str).str.strip().str.upper()
            df_h_sheet = df_h_sheet.sort_values(['Ticker', 'Date of Pay'])
        else:
            df_h_sheet = pd.DataFrame(columns=['Date of Pay', 'Ticker'])
        return df_m, df_h_sheet
    except Exception as e:
        st.error(f"Failed to load Google Sheets: {e}")
        return None, None

@st.cache_data(ttl=3600)
def fetch_single_asset(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max", auto_adjust=False)
        if hist.empty:
            return pd.DataFrame(), pd.DataFrame()
        hist = hist.reset_index()
        prices = hist[['Date', 'Close']].copy().rename(columns={'Close': 'Closing Price'})
        prices['Ticker'] = ticker
        prices['Date'] = pd.to_datetime(prices['Date']).dt.tz_localize(None)
        meta = df_m[df_m['Ticker'] == ticker]
        if not meta.empty:
            for col in ['Strategy', 'Company', 'Underlying']:
                if col in meta.columns:
                    val = meta.iloc[0][col]
                    if pd.isna(val) or str(val).strip() == '' or str(val).lower() == 'nan':
                        prices[col] = '-'
                    else:
                        prices[col] = val
                else:
                    prices[col] = '-'
        prices['Closing Price'] = pd.to_numeric(prices['Closing Price'], errors='coerce').fillna(0.0)
        final_history = []
        if 'Dividends' in hist.columns:
            divs = hist[hist['Dividends'] > 0][['Date', 'Dividends']].copy()
            if not divs.empty:
                divs['YF_Ex_Date'] = pd.to_datetime(divs['Date']).dt.tz_localize(None)
                divs = divs.sort_values('YF_Ex_Date')
                sheet_dates = df_h_sheet[df_h_sheet['Ticker'] == ticker].sort_values('Date of Pay')
                for j, yf_row in enumerate(divs.itertuples()):
                    amt = yf_row.Dividends
                    pay_date = sheet_dates.iloc[j]['Date of Pay'] if j < len(sheet_dates) else yf_row.YF_Ex_Date
                    final_history.append({'Date of Pay': pay_date, 'Ticker': ticker, 'Amount': amt})
        df_h_single = pd.DataFrame(final_history) if final_history else pd.DataFrame(columns=['Date of Pay', 'Amount', 'Ticker'])
        return prices, df_h_single
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def calculate_journey(ticker, start_date, end_date, initial_shares, drip_enabled, unified_df, history_df):
    t_price = unified_df[unified_df['Ticker'] == ticker].sort_values('Date')
    journey = t_price[(t_price['Date'] >= start_date) & (t_price['Date'] <= end_date)].copy()
    if journey.empty:
        return journey
    t_divs = history_df[history_df['Ticker'] == ticker].sort_values('Date of Pay')
    relevant_divs = t_divs[(t_divs['Date of Pay'] >= start_date) & (t_divs['Date of Pay'] <= end_date)].copy()
    journey = journey.set_index('Date')
    journey['Shares'] = initial_shares
    journey['Cash_Pocketed'] = 0.0
    current_shares = initial_shares
    cum_cash = 0.0
    if not relevant_divs.empty:
        for _, row in relevant_divs.iterrows():
            d_date = row['Date of Pay']
            d_amt = row['Amount']
            if d_date in journey.index:
                payout = current_shares * d_amt
                if drip_enabled:
                    reinvest_price = journey.loc[d_date, 'Closing Price']
                    if reinvest_price > 0:
                        new_shares = payout / reinvest_price
                        current_shares += new_shares
                    journey.loc[d_date:, 'Shares'] = current_shares
                else:
                    cum_cash += payout
                    journey.loc[d_date:, 'Cash_Pocketed'] = cum_cash
    journey = journey.reset_index()
    journey['Market_Value'] = journey['Closing Price'] * journey['Shares']
    journey['Base_Asset_Value'] = journey['Closing Price'] * initial_shares
    if drip_enabled:
        journey['True_Value'] = journey['Market_Value']
    else:
        journey['True_Value'] = journey['Market_Value'] + journey['Cash_Pocketed']
    return journey

# --- DATA LOADING (Using terminal's load for consistency) ---
df_m, df_h_sheet = load_base_sheets()
if df_m is None:
    st.stop()
df = df_m.copy()  # For search compatibility
text_cols = ['Ticker', 'Strategy', 'Company', 'Underlying', 'Payout', 'Category', 'Pay Date', 'Declaration Date', 'Ex-Div Date']
for col in text_cols:
    if col in df.columns: df[col] = df[col].fillna('-').astype(str)
    else: df[col] = '-'
for col in ['Dividend', 'Current Price', 'Latest Distribution']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- SEARCH AND FILTER SECTION (From Search App) ---
left_col, right_col = st.columns([2, 1])
with left_col:
    st.text_input("", placeholder="🔍 Search any Ticker, Strategy, Company or Underlying...", key="search_term")
    c1, c2 = st.columns(2)
    with c1:
        all_tags = set()
        if 'Category' in df.columns:
            for tags in df['Category'].str.split(','):
                for tag in tags:
                    if tag.strip() and tag.strip() != '-': all_tags.add(tag.strip())
        selected_strategies = st.multiselect("", options=sorted(list(all_tags)), placeholder="📊 Filter by Strategy")
    with c2:
        freq_opts = sorted(df['Payout'].unique().tolist())
        selected_freq = st.multiselect("", options=freq_opts, placeholder="⏰ Payout Frequency")
with right_col:
    yield_range = st.slider("💰 Search by Annualized Yield %", 0, 150, (0, 150))

# --- FILTER LOGIC ---
search_input = st.session_state.search_term
has_search = bool(search_input)
has_strat = bool(selected_strategies)
has_freq = bool(selected_freq)
has_yield = yield_range[0] > 0 or yield_range[1] < 150
filtered = df.copy()
if has_search or has_strat or has_freq or has_yield:
    if has_search:
        term = search_input.lower()
        filtered = filtered[
            filtered['Ticker'].str.lower().str.contains(term) |
            filtered['Strategy'].str.lower().str.contains(term) |
            filtered['Company'].str.lower().str.contains(term) |
            filtered['Category'].str.lower().str.contains(term) |
            filtered['Underlying'].str.lower().str.contains(term)
        ]
    if has_strat:
        for strat in selected_strategies:
            filtered = filtered[filtered['Category'].str.contains(strat, case=False)]
    if has_freq:
        filtered = filtered[filtered['Payout'].isin(selected_freq)]
    if has_yield and 'Dividend' in filtered.columns:
        filtered = filtered[(filtered['Dividend'] >= yield_range[0]) & (filtered['Dividend'] <= yield_range[1])]
    if not filtered.empty:
        rename_map = {
            'Current Price': 'Price',
            'Dividend': 'Yield %',
            'Latest Distribution': 'Latest Dist',
            'Declaration Date': 'Declaration Date',
            'Ex-Div Date': 'Ex-Div Date',
            'Pay Date': 'Pay Date'
        }
        target_order = [
            'Ticker', 'Strategy', 'Underlying', 'Current Price', 'Payout',
            'Latest Distribution', 'Dividend', 'Declaration Date', 'Ex-Div Date', 'Pay Date'
        ]
        existing_cols = [c for c in target_order if c in filtered.columns]
        display_df = filtered[existing_cols].rename(columns=rename_map)
        if 'Yield %' in display_df.columns:
            display_df = display_df.sort_values(by='Yield %', ascending=False)
        # --- Make Table Selectable with Order Icons ---
        if 'selected_tickers' not in st.session_state:
            st.session_state.selected_tickers = []
        def num_to_emoji(n):
            emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
            return emojis[n-1] if n <= 10 else str(n)
        display_df['Status'] = display_df['Ticker'].apply(lambda x: num_to_emoji(st.session_state.selected_tickers.index(x)+1) if x in st.session_state.selected_tickers else '')
        display_df['Monitor'] = display_df['Ticker'].apply(lambda x: x in st.session_state.selected_tickers)
        num_rows = len(display_df)
        dynamic_height = min((num_rows * 35) + 38, 500)
        edited_df = st.data_editor(
            display_df,
            column_order=['Status', 'Monitor'] + [col for col in display_df.columns if col not in ['Status', 'Monitor']],
            column_config={
                'Status': st.column_config.TextColumn(' ', width='small'),
                'Monitor': st.column_config.CheckboxColumn('Monitor', default=False, width='medium'),
                'Yield %': st.column_config.NumberColumn(format='%.2f%%'),
                'Price': st.column_config.NumberColumn(format='$%.2f'),
                'Latest Dist': st.column_config.NumberColumn(format='$%.4f')
            },
            disabled=[col for col in display_df.columns if col != 'Monitor'],
            hide_index=True,
            use_container_width=True,
            height=dynamic_height
        )
        current_selected = edited_df[edited_df['Monitor']]['Ticker'].tolist()
        new_added = [t for t in current_selected if t not in st.session_state.selected_tickers]
        st.session_state.selected_tickers += new_added
        st.session_state.selected_tickers = [t for t in st.session_state.selected_tickers if t in current_selected]
    else:
        st.info("No funds match your criteria.")

# --- CHARTS SECTION (Only if tickers selected) ---
selected_tickers = st.session_state.get('selected_tickers', [])
if selected_tickers:
    df_m, df_h_sheet = load_base_sheets()  # Reload if needed

    # --- Single Asset Section ---
    if len(selected_tickers) >= 1:
        st.header("Single Asset Performance")
        # Filters above chart (from sidebar, now in columns)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            selected_ticker = st.selectbox("View Chart for", selected_tickers)
        with col2:
            use_inception = st.checkbox("🚀 Start from Inception", value=False)
        with col3:
            date_mode = st.radio("Simulation End", ["Hold to Present", "Sell on Specific Date"])
        with col4:
            mode = st.radio("Input Method", ["Share Count", "Dollar Amount"])
        with col5:
            use_drip = st.checkbox("🔄 Enable DRIP", value=False)
        with col6:
            overlay_underlyings = st.checkbox("📊 Overlay Underlying", value=False)
        # Fetch data
        price_df, hist_df = fetch_single_asset(selected_ticker)
        if price_df.empty:
            st.error("No data found for this ticker.")
        else:
            inception_date = price_df['Date'].min()
            if use_inception:
                buy_date = inception_date
            else:
                default_date = pd.to_datetime("today") - pd.DateOffset(months=12)
                if default_date < inception_date: default_date = inception_date
                buy_date = col2.date_input("Purchase Date", default_date, label_visibility="collapsed")
            buy_date = pd.to_datetime(buy_date)
            end_date = pd.to_datetime("today") if date_mode == "Hold to Present" else pd.to_datetime(col3.date_input("Sell Date", pd.to_datetime("today"), label_visibility="collapsed"))
            temp_journey = price_df[(price_df['Date'] >= buy_date) & (price_df['Date'] <= end_date)]
            if not temp_journey.empty:
                entry_price = temp_journey.iloc[0]['Closing Price']
                if mode == "Share Count":
                    initial_shares = col4.number_input("Shares Owned", min_value=1, value=10, label_visibility="collapsed")
                else:
                    dollars = col4.number_input("Amount Invested ($)", min_value=100, value=1000, step=100, label_visibility="collapsed")
                    initial_shares = dollars / entry_price if entry_price > 0 else 0
            else:
                st.error("No data for date range.")
                return
            # Calculate and display
            journey = calculate_journey(selected_ticker, buy_date, end_date, initial_shares, use_drip, price_df, hist_df)
            if journey.empty:
                st.error("Journey calculation failed.")
            else:
                initial_cap = entry_price * initial_shares
                current_market_val = journey.iloc[-1]['Market_Value']
                cash_total = journey.iloc[-1]['Cash_Pocketed']
                current_total_val = journey.iloc[-1]['True_Value']
                final_shares = journey.iloc[-1]['Shares']
                market_pl = current_market_val - initial_cap
                market_pl_pct = (market_pl / initial_cap) * 100 if initial_cap != 0 else 0
                total_pl = current_total_val - initial_cap
                total_return_pct = (total_pl / initial_cap) * 100 if initial_cap != 0 else 0
                days_held = (end_date - buy_date).days
                annual_yield = (cash_total / initial_cap) * (365.25 / days_held) * 100 if days_held > 0 and initial_cap > 0 else 0
                # Header and Meta
                meta_row = price_df.iloc[0]
                asset_strategy = meta_row.get('Strategy', '-')
                asset_company = meta_row.get('Company', '-')
                und = meta_row.get('Underlying', '-')
                if pd.isna(und) or str(und).strip() == '' or str(und).lower() == 'nan':
                    und = '-'
                und_pct = None
                if und != '-':
                    df_u_und, df_h_und = fetch_overlay_data(und, buy_date, end_date)
                    if df_u_und is not None and not df_u_und.empty:
                        t_price_check = df_u_und[(df_u_und['Date'] >= buy_date) & (df_u_und['Date'] <= end_date)].sort_values('Date')
                        if not t_price_check.empty:
                            start_p = t_price_check.iloc[0]['Closing Price']
                            t_journey = calculate_journey(und, buy_date, end_date, 1.0, use_drip, df_u_und, df_h_und)
                            if not t_journey.empty:
                                final_true_val = t_journey.iloc[-1]['True_Value']
                                und_pct = ((final_true_val - start_p) / start_p) * 100 if start_p > 0 else 0
                col_head, col_meta = st.columns([1.8, 1.2])
                with col_head:
                    st.markdown(f"""
                        <div style="margin-top: -10px;">
                            <h1 style="font-size: 2.5rem; margin-bottom: 0px; color: #E6EDF3; line-height: 1.2;">
                                Performance Simulator : <span style="color: #8AC7DE;">{selected_ticker}</span>
                            </h1>
                            <p style="font-size: 1.1rem; color: #8AC7DE; opacity: 0.8; margin-top: -5px; margin-bottom: 10px;">
                                <b>{final_shares:.2f} shares</b> &nbsp;|&nbsp; {buy_date.date()} ➝ {end_date.date()} ({days_held} days)
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                with col_meta:
                    if und == '-' or und_pct is None:
                        meta_cols = st.columns(2)
                        with meta_cols[0]:
                            st.markdown(f"""
                                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px; text-align: center; height: auto; min-height: 80px; overflow: auto;">
                                    <div style="color: #8AC7DE; font-size: 0.7rem; text-transform: uppercase; white-space: normal; word-break: break-word;">Strategy</div>
                                    <div style="color: white; font-size: 1.0rem; font-weight: 600; margin-top: 4px; word-break: break-word; overflow-wrap: break-word; white-space: normal; hyphens: auto; line-height: 1.2; padding-bottom: 8px;">{asset_strategy}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with meta_cols[1]:
                            st.markdown(f"""
                                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px; text-align: center; height: auto; min-height: 80px; overflow: auto;">
                                    <div style="color: #8AC7DE; font-size: 0.7rem; text-transform: uppercase; white-space: normal; word-break: break-word;">Company</div>
                                    <div style="color: white; font-size: 1.0rem; font-weight: 600; margin-top: 4px; word-break: break-word; overflow-wrap: break-word; white-space: normal; hyphens: auto; line-height: 1.2; padding-bottom: 8px;">{asset_company}</div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        meta_cols = st.columns(3)
                        with meta_cols[0]:
                            st.markdown(f"""
                                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px; text-align: center; height: auto; min-height: 80px; overflow: auto;">
                                    <div style="color: #8AC7DE; font-size: 0.7rem; text-transform: uppercase; white-space: normal; word-break: break-word;">Strategy</div>
                                    <div style="color: white; font-size: 1.0rem; font-weight: 600; margin-top: 4px; word-break: break-word; overflow-wrap: break-word; white-space: normal; hyphens: auto; line-height: 1.2; padding-bottom: 8px;">{asset_strategy}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with meta_cols[1]:
                            st.markdown(f"""
                                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px; text-align: center; height: auto; min-height: 80px; overflow: auto;">
                                    <div style="color: #8AC7DE; font-size: 0.7rem; text-transform: uppercase; white-space: normal; word-break: break-word;">Company</div>
                                    <div style="color: white; font-size: 1.0rem; font-weight: 600; margin-top: 4px; word-break: break-word; overflow-wrap: break-word; white-space: normal; hyphens: auto; line-height: 1.2; padding-bottom: 8px;">{asset_company}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with meta_cols[2]:
                            und_color = "#00C805" if und_pct >= 0 else "#FF4B4B"
                            symbol = "▲" if und_pct > 0 else "▼" if und_pct < 0 else ""
                            st.markdown(f"""
                                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px; text-align: center; height: auto; min-height: 80px; overflow: auto;">
                                    <div style="color: #8AC7DE; font-size: 0.7rem; text-transform: uppercase; white-space: normal; word-break: break-word;">Underlying</div>
                                    <div style="color: white; font-size: 1.0rem; font-weight: 600; margin-top: 4px; word-break: break-word; overflow-wrap: break-word; white-space: normal; hyphens: auto; line-height: 1.2; padding-bottom: 8px;">
                                        {und} <span style="color: {und_color}; filter: brightness(1.2);">{symbol} {und_pct:+.2f}%</span> <span title="this is the percentage gain or loss of the underlying ticker over the timeline you have selected" style="cursor: help; color: #8AC7DE; font-size: 0.8rem;">?</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                # Metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Initial Capital", f"${initial_cap:,.2f}")
                val_label = "End Asset Value" if not use_drip else "End Value (DRIP)"
                cash_label = "Dividends Collected" if not use_drip else "New Shares Acquired"
                m2.metric(val_label, f"${current_market_val:,.2f}", f"{market_pl_pct:+.2f}%")
                if use_drip:
                    shares_gained = final_shares - initial_shares
                    m3.metric(cash_label, f"{shares_gained:.2f} Shares")
                    m4.metric("Effective Yield", "N/A (Reinvested)")
                else:
                    m3.metric(cash_label, f"${cash_total:,.2f}")
                    m4.metric("Annualized Yield", f"{annual_yield:.2f}%")
                m5.metric("True Total Value", f"${current_total_val:,.2f}", f"{total_return_pct:.2f}%")
                # Chart
                fig = go.Figure()
                bottom_y = journey['Market_Value'] if not use_drip else journey['Base_Asset_Value']
                top_y = journey['True_Value']
                price_color = '#8AC7DE' if journey.iloc[-1]['Closing Price'] >= journey.iloc[0]['Closing Price'] else '#FF4B4B'
                fig.add_trace(go.Scatter(x=journey['Date'], y=bottom_y, mode='lines', name='Asset Price', line=dict(color=price_color, width=2)))
                fig.add_trace(go.Scatter(x=journey['Date'], y=top_y, mode='lines', name='True Value', line=dict(color='#00C805', width=3), fill='tonexty', fillcolor='rgba(0, 200, 5, 0.1)'))
                fig.add_hline(y=initial_cap, line_dash="dash", line_color="white", opacity=0.3)
                profit_text = f"PROFIT: +${total_pl:,.2f}" if total_pl >= 0 else f"LOSS: -${abs(total_pl):,.2f}"
                profit_bg = "#00C805" if total_pl >= 0 else "#FF4B4B"
                fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper", text=profit_text, showarrow=False, font=dict(family="Arial Black, sans-serif", size=16, color="white"), bgcolor=profit_bg, bordercolor=profit_bg, borderpad=8, opacity=0.9, align="left")
                if overlay_underlyings and und != '-':
                    df_u_und, df_h_und = fetch_overlay_data(und, buy_date, end_date)
                    if df_u_und is not None:
                        t_price_check = df_u_und[(df_u_und['Date'] >= buy_date) & (df_u_und['Date'] <= end_date)].sort_values('Date')
                        if not t_price_check.empty:
                            initial_s = initial_cap / t_price_check.iloc[0]['Closing Price']
                            t_journey = calculate_journey(und, buy_date, end_date, initial_s, use_drip, df_u_und, df_h_und)
                            if not t_journey.empty:
                                fig.add_trace(go.Scatter(x=t_journey['Date'], y=t_journey['True_Value'], mode='lines', name=und, line=dict(color='#FFD700', width=2, dash='dash')))
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=340, margin=dict(l=0, r=0, t=20, b=0), showlegend=False, hovermode="x unified", xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown("""
                    <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 5px 8px; text-align: center;">
                        <span style="color: #00C805; font-weight: 800;">💚 True Value (Total Equity)</span> &nbsp;&nbsp;
                        <span style="color: #8AC7DE; font-weight: 800;">🔵 Price Appreciation</span> &nbsp;&nbsp;
                        <span style="color: #FF4B4B; font-weight: 800;">🔴 Price Erosion</span>
                    </div>
                """, unsafe_allow_html=True)
                with st.expander("View Data"):
                    st.dataframe(journey.sort_values('Date', ascending=False), use_container_width=True)

    # --- Head-to-Head Section (If multiple selected) ---
    if len(selected_tickers) > 1:
        st.header("Head-to-Head Comparison")
        # Filters above chart
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            default_start = pd.to_datetime("today") - pd.DateOffset(months=12)
            buy_date = st.date_input("Start Date", default_start)
            buy_date = pd.to_datetime(buy_date)
        with col2:
            end_date = st.date_input("End Date", pd.to_datetime("today"))
            end_date = pd.to_datetime(end_date)
        with col3:
            sim_amt = st.number_input("Hypothetical Investment ($)", value=10000, step=1000)
        with col4:
            use_drip = st.checkbox("🔄 Enable DRIP", value=False)
        with col5:
            overlay_underlyings = st.checkbox("📊 Overlay Underlyings", value=False)
        # Logic
        comp_data = []
        fig_comp = go.Figure()
        colors = ['#00C805', '#F59E0B', '#8AC7DE', '#FF4B4B', '#A855F7', '#EC4899', '#EAB308']
        for idx, t in enumerate(selected_tickers):
            price_df, hist_df = fetch_single_asset(t)
            if price_df.empty: continue
            t_price_check = price_df[(price_df['Date'] >= buy_date) & (price_df['Date'] <= end_date)].sort_values('Date')
            if t_price_check.empty: continue
            initial_s = sim_amt / t_price_check.iloc[0]['Closing Price']
            t_journey = calculate_journey(t, buy_date, end_date, initial_s, use_drip, price_df, hist_df)
            if t_journey.empty: continue
            initial_cap = sim_amt
            t_journey['Total_Return_Pct'] = ((t_journey['True_Value'] - initial_cap) / initial_cap) * 100
            line_color = colors[idx % len(colors)]
            fig_comp.add_trace(go.Scatter(x=t_journey['Date'], y=t_journey['Total_Return_Pct'], mode='lines', name=t, line=dict(color=line_color, width=3)))
            final_row = t_journey.iloc[-1]
            final_ret = final_row['Total_Return_Pct']
            end_value = final_row['Market_Value']
            cash_generated = final_row['Cash_Pocketed']
            final_total = final_row['True_Value']
            if not use_drip:
                yield_pct = (cash_generated / initial_cap) * 100
            else:
                yield_pct = 0.0
            shares_added = final_row['Shares'] - initial_s
            data_row = {"Ticker": t, "Total Return": final_ret, "💚 Total Value": final_total}
            if use_drip:
                data_row["📈 New Shares Added"] = shares_added
                data_row["Yield %"] = "N/A"
            else:
                data_row["💰 Cash Generated"] = cash_generated
                data_row["Yield %"] = yield_pct
                data_row["📉 Share Value (Remaining)"] = end_value
            comp_data.append(data_row)
        if overlay_underlyings:
            unique_underlyings = set()
            for t in selected_tickers:
                meta = df_m[df_m['Ticker'] == t]
                if not meta.empty:
                    und = meta.iloc[0].get('Underlying', '-')
                    if und != '-' and pd.notna(und) and str(und).strip() != '':
                        unique_underlyings.add(und)
            overlay_colors = ['#00FFFF', '#FF69B4', '#00FF7F', '#87CEEB', '#FFA07A']
            overlay_idx = 0
            for und in unique_underlyings:
                df_u_und, df_h_und = fetch_overlay_data(und, buy_date, end_date)
                if df_u_und is None: continue
                t_price_check = df_u_und[(df_u_und['Date'] >= buy_date) & (df_u_und['Date'] <= end_date)].sort_values('Date')
                if t_price_check.empty: continue
                t_journey = calculate_journey(und, buy_date, end_date, sim_amt / t_price_check.iloc[0]['Closing Price'], use_drip, df_u_und, df_h_und)
                if t_journey.empty: continue
                t_journey['Total_Return_Pct'] = ((t_journey['True_Value'] - sim_amt) / sim_amt) * 100
                fig_comp.add_trace(go.Scatter(x=t_journey['Date'], y=t_journey['Total_Return_Pct'], mode='lines', name=und, line=dict(color=overlay_colors[overlay_idx % len(overlay_colors)], width=2, dash='dash')))
                overlay_idx += 1
                final_row = t_journey.iloc[-1]
                data_row = {"Ticker": und, "Total Return": final_row['Total_Return_Pct'], "💚 Total Value": final_row['True_Value']}
                if use_drip:
                    data_row["📈 New Shares Added"] = final_row['Shares'] - (sim_amt / t_price_check.iloc[0]['Closing Price'])
                    data_row["Yield %"] = "N/A"
                else:
                    data_row["💰 Cash Generated"] = final_row['Cash_Pocketed']
                    data_row["Yield %"] = (final_row['Cash_Pocketed'] / sim_amt) * 100
                    data_row["📉 Share Value (Remaining)"] = final_row['Market_Value']
                comp_data.append(data_row)
        fig_comp.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.5, annotation_text="Break Even")
        fig_comp.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white")), yaxis_title="Total Return (%)", xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})
        if comp_data:
            st.markdown(f"### 🏆 Leaderboard (${sim_amt:,.0f} Investment)")
            df_comp = pd.DataFrame(comp_data).sort_values("Total Return", ascending=False)
            df_display = df_comp.copy()
            df_display['Total Return'] = df_display['Total Return'].apply(lambda x: f"{x:+.2f}%")
            df_display['💚 Total Value'] = df_display['💚 Total Value'].apply(lambda x: f"${x:,.2f}")
            if use_drip:
                df_display['📈 New Shares Added'] = df_display['📈 New Shares Added'].apply(lambda x: f"{x:.2f}")
                cols = ["Ticker", "Total Return", "📈 New Shares Added", "💚 Total Value"]
            else:
                df_display['Yield %'] = df_display['Yield %'].apply(lambda x: f"{x:.2f}%")
                df_display['💰 Cash Generated'] = df_display['💰 Cash Generated'].apply(lambda x: f"${x:,.2f}")
                df_display['📉 Share Value (Remaining)'] = df_display['📉 Share Value (Remaining)'].apply(lambda x: f"${x:,.2f}")
                cols = ["Ticker", "Total Return", "Yield %", "💰 Cash Generated", "📉 Share Value (Remaining)", "💚 Total Value"]
            st.dataframe(df_display[cols], hide_index=True, use_container_width=True)
```
