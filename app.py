import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RIR Integrated Terminal", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 2. THE MEGA-CSS BLOCK (Combined from both apps) ---
st.markdown("""
    <style>
        /* --- MAIN THEME --- */
        .stApp { background-color: #0D1117; color: #E6EDF3; }
        h1, h2, h3, h4, h5, h6, p, label, span { color: #E6EDF3 !important; }

        /* --- INPUTS & CONTROLS --- */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div, 
        div[data-baseweb="base-input"],
        div[data-testid="stDateInput"] > div,
        div[data-testid="stNumberInput"] > div {
            background-color: #1E293B !important;
            border-color: #30363d !important;
            color: #E6EDF3 !important;
            border-radius: 6px !important;
        }
        input { color: #E6EDF3 !important; font-weight: bold !important; }
        
        /* --- DATAFRAME STYLING --- */
        div[data-testid="stDataFrame"] {
            background-color: #0D1117 !important;
            border: 1px solid #30363d;
            border-radius: 5px;
        }
        div[data-testid="stDataFrame"] div[role="columnheader"] {
            background-color: #1E293B !important;
            color: #8AC7DE !important;
            font-weight: bold;
            border-bottom: 1px solid #30363d;
        }
        div[data-testid="stDataFrame"] div[role="gridcell"] {
            background-color: #0D1117 !important;
            color: #E6EDF3 !important;
            border-bottom: 1px solid #30363d;
        }
        
        /* --- YIELD SLIDER CUSTOMIZATION (From Searchbar App) --- */
        div[data-testid="stSlider"] label { color: #E6EDF3 !important; font-weight: 600 !important; }
        div[data-baseweb="slider"] div[style*="background-color: rgb(211, 211, 211)"] { background-color: #4B5563 !important; } /* Track */
        div[data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"] { background-color: #8AC7DE !important; } /* Selected */
        div[role="slider"] { background-color: #8AC7DE !important; border: 2px solid #E6EDF3 !important; }
        
        /* --- METRIC CARDS (From Terminal App) --- */
        div[data-testid="stMetric"] {
            background-color: #1E293B;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            min-height: 80px;
        }
        div[data-testid="stMetricLabel"] p { color: #8AC7DE !important; font-size: 0.85rem !important; font-weight: 600 !important; }
        div[data-testid="stMetricValue"] div { color: #FFFFFF !important; font-size: 1.5rem !important; font-weight: 700 !important; }
        
        /* Gold Highlight for Yield Metric (4th Column usually) */
        div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetric"] {
            border: 1px solid #F59E0B !important;
        }
        /* Green Highlight for Total Value Metric (5th Column usually) */
        div[data-testid="column"]:nth-of-type(5) div[data-testid="stMetric"] {
            background-color: #0D1117 !important;
            border: 2px solid #00C805 !important;
        }

        /* --- EXPANDER STYLING --- */
        .streamlit-expanderHeader {
            background-color: #1E293B !important;
            color: #E6EDF3 !important;
            border: 1px solid #30363d !important;
        }

        /* --- CLEANUP --- */
        .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 98%; }
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. DATA & LOGIC ENGINE ---

@st.cache_data(ttl=600)
def load_data_master():
    """Loads the Master Sheet for the Search Table"""
    csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTKjBzr7QuJk9g7TR6p0-_GdPQDvesG9a1KTny6y5IyK0Z-G0_C98T-AfUyaAdyDB11h3vdpgc_h3Hh/pub?gid=618318322&single=true&output=csv"
    try:
        df = pd.read_csv(csv_url)
    except:
        return pd.DataFrame()
    
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Ticker'])
    df = df[df['Ticker'] != 'Ticker']
    
    # Clean Text
    text_cols = ['Ticker', 'Strategy', 'Company', 'Underlying', 'Payout', 'Category']
    for col in text_cols:
        if col in df.columns: df[col] = df[col].fillna('-').astype(str)
    
    # Clean Numbers
    for col in ['Dividend', 'Current Price']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

@st.cache_data(ttl=3600)
def fetch_ticker_data(ticker):
    """Fetches Price AND Dividends properly from Yahoo Finance"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max", auto_adjust=False)
        if hist.empty: return pd.DataFrame(), pd.DataFrame()
        
        hist = hist.reset_index()
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
        
        prices = hist[['Date', 'Close']].rename(columns={'Close': 'Closing Price'})
        prices['Ticker'] = ticker
        
        divs = hist[hist['Dividends'] > 0][['Date', 'Dividends']].rename(columns={'Date': 'Date of Pay', 'Dividends': 'Amount'})
        divs['Ticker'] = ticker
        
        return prices, divs
    except:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_overlay_data(ticker, start_date, end_date):
    """Fetches underlying asset data for comparison (e.g. TSLA for TSLY)"""
    try:
        t = yf.Ticker(ticker)
        # Fetch with buffer
        hist = t.history(start=start_date, end=end_date + pd.Timedelta(days=5), auto_adjust=False)
        if hist.empty: return None, None
        
        hist = hist.reset_index()
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
        
        prices = hist[['Date', 'Close']].rename(columns={'Close': 'Closing Price'})
        prices['Ticker'] = ticker
        divs = hist[hist['Dividends'] > 0][['Date', 'Dividends']].rename(columns={'Date': 'Date of Pay', 'Dividends': 'Amount'})
        
        return prices, divs
    except:
        return None, None

def calculate_journey(ticker, start_date, end_date, initial_shares, drip_enabled, price_df, div_df):
    """The Full Compounding Engine"""
    # Filter Data
    journey = price_df[(price_df['Date'] >= start_date) & (price_df['Date'] <= end_date)].sort_values('Date').copy()
    if journey.empty: return pd.DataFrame()
    
    # Get Divs in range
    relevant_divs = div_df[(div_df['Date of Pay'] >= start_date) & (div_df['Date of Pay'] <= end_date)].sort_values('Date of Pay').copy()
    
    journey = journey.set_index('Date')
    journey['Shares'] = initial_shares
    journey['Cash_Pocketed'] = 0.0
    
    current_shares = initial_shares
    cum_cash = 0.0
    
    if not relevant_divs.empty:
        for _, row in relevant_divs.iterrows():
            d_date = row['Date of Pay']
            d_amt = row['Amount']
            
            # Logic: Apply dividend to the date it hit (or closest forward date in price history)
            if d_date in journey.index:
                payout = current_shares * d_amt
                
                if drip_enabled:
                    reinvest_price = journey.loc[d_date, 'Closing Price']
                    if reinvest_price > 0:
                        new_shares = payout / reinvest_price
                        current_shares += new_shares
                    # Update all future rows
                    journey.loc[d_date:, 'Shares'] = current_shares
                else:
                    cum_cash += payout
                    journey.loc[d_date:, 'Cash_Pocketed'] = cum_cash

    journey = journey.reset_index()
    journey['Market_Value'] = journey['Closing Price'] * journey['Shares']
    
    if drip_enabled:
        journey['True_Value'] = journey['Market_Value']
    else:
        journey['True_Value'] = journey['Market_Value'] + journey['Cash_Pocketed']
        
    return journey

# --- 4. TOP SECTION: SEARCH BAR APP (Restored) ---

df_master = load_data_master()

st.markdown("### 🔎 RIR High-Yield Explorer")
col_search, col_slider = st.columns([2, 1])

with col_search:
    search_term = st.text_input("", placeholder="🔍 Search Ticker, Strategy, Company...", key="search")
    f1, f2 = st.columns(2)
    with f1:
        all_cats = set()
        if 'Category' in df_master.columns:
            for x in df_master['Category'].dropna():
                for tag in x.split(','):
                    if tag.strip() != '-': all_cats.add(tag.strip())
        sel_strat = st.multiselect("", options=sorted(list(all_cats)), placeholder="📊 Filter Strategy")
    with f2:
        freqs = sorted(df_master['Payout'].unique().tolist()) if 'Payout' in df_master.columns else []
        sel_freq = st.multiselect("", options=freqs, placeholder="⏰ Payout Frequency")

with col_slider:
    sel_yield = st.slider("💰 Annualized Yield %", 0, 150, (0, 150))

# Filtering Logic
filtered = df_master.copy()
if search_term:
    t = search_term.lower()
    filtered = filtered[
        filtered['Ticker'].str.lower().str.contains(t) | 
        filtered['Strategy'].str.lower().str.contains(t) | 
        filtered['Company'].str.lower().str.contains(t)
    ]
if sel_strat:
    for s in sel_strat:
        filtered = filtered[filtered['Category'].str.contains(s, case=False)]
if sel_freq:
    filtered = filtered[filtered['Payout'].isin(sel_freq)]
if 'Dividend' in filtered.columns:
    filtered = filtered[(filtered['Dividend'] >= sel_yield[0]) & (filtered['Dividend'] <= sel_yield[1])]

# Interactive Table
if not filtered.empty:
    st.markdown("⬇️ **Select Tickers to Monitor.** (First selection = Primary Chart)")
    disp_df = filtered[['Ticker', 'Strategy', 'Current Price', 'Dividend', 'Payout']].rename(
        columns={'Dividend': 'Yield %', 'Current Price': 'Price'}
    )
    selection = st.dataframe(
        disp_df.style.format({'Yield %': '{:.2f}%', 'Price': '${:.2f}'}),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row"
    )
    selected_rows = filtered.iloc[selection.selection['rows']]
else:
    st.info("No funds match your filters.")
    selected_rows = pd.DataFrame()

st.divider()

# --- 5. BOTTOM SECTION: HIGH YIELD TERMINAL (Fully Restored) ---

if selected_rows.empty:
    st.info("👆 Select a ticker above to activate the Terminal.")
else:
    primary_ticker = selected_rows.iloc[0]['Ticker']
    compare_tickers = selected_rows['Ticker'].tolist()
    
    # --- CONTROL DECK (Restored Logic) ---
    with st.container():
        st.markdown(f"### ⚙️ Terminal Controls: {primary_ticker}")
        
        # Row 1: Dates & Inception
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_inception = st.checkbox("🚀 Start from Inception", value=False)
        with c2:
            date_mode = st.radio("End Mode", ["Hold to Present", "Sell Specific Date"], horizontal=True, label_visibility="collapsed")
        
        # Row 2: Inputs
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            # Date Logic
            today = pd.to_datetime("today")
            if use_inception:
                # Placeholder, will update after data fetch
                start_date = today 
            else:
                start_date = st.date_input("Start Date", today - pd.DateOffset(months=12))
            start_date = pd.to_datetime(start_date)
            
        with i2:
            if date_mode == "Sell Specific Date":
                end_date = st.date_input("Sell Date", today)
            else:
                end_date = today
            end_date = pd.to_datetime(end_date)
            
        with i3:
            input_method = st.radio("Input Method", ["$ Amount", "Share Count"], horizontal=True)
            if input_method == "$ Amount":
                invest_val = st.number_input("Investment ($)", value=10000, step=1000)
            else:
                share_count_val = st.number_input("Share Count", value=1000, step=100)
                
        with i4:
            st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True) # Spacer
            c_drip, c_und = st.columns(2)
            with c_drip: use_drip = st.checkbox("🔄 DRIP", value=False)
            with c_und: show_underlying = st.checkbox("📊 Underlying", value=False)

    # --- DATA FETCHING ---
    with st.spinner(f"Crunching numbers for {primary_ticker}..."):
        p_df, d_df = fetch_ticker_data(primary_ticker)
        
        # Underlying Fetch
        meta = df_master[df_master['Ticker'] == primary_ticker].iloc[0]
        underlying_ticker = meta.get('Underlying', '-')
        
        u_p_df, u_d_df = None, None
        if show_underlying and underlying_ticker not in ['-', 'nan']:
            u_p_df, u_d_df = fetch_overlay_data(underlying_ticker, start_date if not use_inception else p_df['Date'].min(), end_date)

    if p_df.empty:
        st.error(f"No data found for {primary_ticker}")
    else:
        # Resolve Inception Date
        data_min_date = p_df['Date'].min()
        if use_inception:
            valid_start = data_min_date
            st.caption(f"ℹ️ Starting from inception date: {valid_start.date()}")
        else:
            valid_start = max(start_date, data_min_date)

        # Resolve Shares vs Dollars
        start_price_row = p_df.loc[p_df['Date'] >= valid_start]
        if start_price_row.empty:
            st.error("Start date is after the last available data point.")
        else:
            start_price = start_price_row.iloc[0]['Closing Price']
            
            if input_method == "$ Amount":
                initial_shares = invest_val / start_price
                initial_capital = invest_val
            else:
                initial_shares = share_count_val
                initial_capital = share_count_val * start_price

            # CALCULATION JOURNEY
            journey = calculate_journey(primary_ticker, valid_start, end_date, initial_shares, use_drip, p_df, d_df)
            
            if journey.empty:
                st.error("Journey calculation failed.")
            else:
                # --- METRICS SECTION (Restored) ---
                final = journey.iloc[-1]
                
                # Underlying Perf String
                und_str = ""
                if u_p_df is not None:
                    try:
                        us_p = u_p_df[u_p_df['Date'] >= valid_start].iloc[0]['Closing Price']
                        ue_p = u_p_df[u_p_df['Date'] <= end_date].iloc[-1]['Closing Price']
                        u_pct = ((ue_p - us_p)/us_p)*100
                        col = "#00C805" if u_pct >= 0 else "#FF4B4B"
                        und_str = f"vs Underlying: {underlying_ticker} <span style='color:{col}'>{u_pct:+.2f}%</span>"
                    except: pass

                st.markdown(f"""
                    <div style="margin-top: 10px; margin-bottom: 15px;">
                        <h1 style="margin:0; font-size: 2.2rem;">{primary_ticker} <span style="font-size: 1rem; color: #8AC7DE; font-weight: normal;">{meta['Strategy']}</span></h1>
                        <div style="font-size: 0.9rem; color: #94A3B8;">{und_str}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                m1, m2, m3, m4, m5 = st.columns(5)
                
                m1.metric("Initial Capital", f"${initial_capital:,.0f}")
                
                curr_val = final['Market_Value']
                val_chg = ((curr_val - initial_capital)/initial_capital)*100
                m2.metric("End Asset Value", f"${curr_val:,.2f}", f"{val_chg:+.2f}%")
                
                if use_drip:
                    added = final['Shares'] - initial_shares
                    m3.metric("Shares Added", f"{added:.2f}")
                    m4.metric("Effective Yield", "N/A (DRIP)")
                else:
                    cash = final['Cash_Pocketed']
                    m3.metric("Cash Generated", f"${cash:,.2f}")
                    days = (end_date - valid_start).days
                    ann_y = (cash/initial_capital)*(365/days)*100 if days > 0 else 0
                    m4.metric("Annualized Yield", f"{ann_y:.2f}%")
                
                true_val = final['True_Value']
                tot_ret = ((true_val - initial_capital)/initial_capital)*100
                m5.metric("True Total Value", f"${true_val:,.2f}", f"{tot_ret:+.2f}%")
                
                # --- CHART SECTION ---
                fig = go.Figure()
                
                # Asset Line
                p_col = '#8AC7DE' if final['Closing Price'] >= start_price else '#FF4B4B'
                fig.add_trace(go.Scatter(x=journey['Date'], y=journey['Market_Value'], mode='lines', name='Asset Value', line=dict(color=p_col, width=2)))
                
                # True Value Line
                fig.add_trace(go.Scatter(x=journey['Date'], y=journey['True_Value'], mode='lines', name='True Value', line=dict(color='#00C805', width=3), fill='tonexty', fillcolor='rgba(0, 200, 5, 0.1)'))
                
                # Underlying Line
                if u_p_df is not None:
                    try:
                        u_start_p = u_p_df.loc[u_p_df['Date'] >= valid_start].iloc[0]['Closing Price']
                        # Hypothetical: If we bought underlying with same start capital
                        u_shares = initial_capital / u_start_p
                        u_j = calculate_journey(underlying_ticker, valid_start, end_date, u_shares, use_drip, u_p_df, u_d_df)
                        fig.add_trace(go.Scatter(x=u_j['Date'], y=u_j['True_Value'], mode='lines', name=f"{underlying_ticker} (Underlying)", line=dict(color='#F59E0B', width=2, dash='dash')))
                    except: pass
                
                fig.add_hline(y=initial_capital, line_dash="dash", line_color="white", opacity=0.3)
                
                # Annotation
                pl = true_val - initial_capital
                txt = f"+${pl:,.2f}" if pl >= 0 else f"-${abs(pl):,.2f}"
                bg = "#00C805" if pl >= 0 else "#FF4B4B"
                fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper", text=f"P/L: {txt}", showarrow=False, bgcolor=bg, bordercolor=bg, font=dict(color="white"))
                
                fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                # --- VIEW DATA EXPANDER (Restored) ---
                with st.expander("📂 View Raw Data Calculation"):
                    st.dataframe(journey.sort_values('Date', ascending=False), use_container_width=True)

    # --- 6. HEAD-TO-HEAD (Restored) ---
    if len(compare_tickers) > 1:
        st.divider()
        st.markdown("### ⚔️ Head-to-Head Comparison")
        
        comp_res = []
        fig_c = go.Figure()
        colors = ['#00C805', '#F59E0B', '#8AC7DE', '#FF4B4B', '#A855F7']
        
        for idx, t in enumerate(compare_tickers):
            tp, td = fetch_ticker_data(t)
            if tp.empty: continue
            
            tv_start = max(start_date, tp['Date'].min()) if not use_inception else tp['Date'].min()
            
            try:
                ts_p = tp.loc[tp['Date'] >= tv_start].iloc[0]['Closing Price']
                
                # Normalize investment for comparison (Always assume $10k for fair comparison or use current input)
                # Let's use the user's input capital to be consistent
                t_shares = initial_capital / ts_p
                
                tj = calculate_journey(t, tv_start, end_date, t_shares, use_drip, tp, td)
                
                if not tj.empty:
                    tj['Pct'] = ((tj['True_Value'] - initial_capital)/initial_capital)*100
                    
                    # Style
                    is_prim = (t == primary_ticker)
                    w = 4 if is_prim else 2
                    op = 1.0 if is_prim else 0.6
                    col = colors[idx % len(colors)]
                    
                    fig_c.add_trace(go.Scatter(x=tj['Date'], y=tj['Pct'], mode='lines', name=t, line=dict(color=col, width=w), opacity=op))
                    
                    last = tj.iloc[-1]
                    rec = {"Ticker": t, "Total Return": last['Pct'], "True Value": last['True_Value']}
                    if not use_drip: rec["Cash Gen"] = last['Cash_Pocketed']
                    comp_res.append(rec)
            except: continue
            
        fig_c.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3)
        fig_c.update_layout(template="plotly_dark", height=400, yaxis_title="Total Return (%)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
        st.plotly_chart(fig_c, use_container_width=True)
        
        if comp_res:
            lb = pd.DataFrame(comp_res).sort_values("Total Return", ascending=False)
            lb['Total Return'] = lb['Total Return'].apply(lambda x: f"{x:+.2f}%")
            lb['True Value'] = lb['True Value'].apply(lambda x: f"${x:,.2f}")
            if "Cash Gen" in lb.columns: lb['Cash Gen'] = lb['Cash Gen'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(lb, hide_index=True, use_container_width=True)
