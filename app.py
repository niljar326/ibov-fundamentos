import streamlit as st
import streamlit.components.v1 as components 

# --- 1. CONFIGURAÇÃO DA PÁGINA (DEVE SER A PRIMEIRA COISA) ---
st.set_page_config(
    page_title="Melhores Ações Ibovespa 2025 | Ranking Fundamentalista e Dividendos",
    layout="wide",
    page_icon="🇧🇷"
)

# --- 2. GERENCIAMENTO DE ESTADO (CORREÇÃO PARA NUVEM) ---
# Inicializa as variáveis com nomes novos para forçar o bloqueio no servidor
if 'access_key_tab1_vFinal' not in st.session_state:
    st.session_state.access_key_tab1_vFinal = False

if 'access_key_tab3_vFinal' not in st.session_state:
    st.session_state.access_key_tab3_vFinal = False

if 'tv_symbol' not in st.session_state: 
    st.session_state.tv_symbol = "BMFBOVESPA:LREN3"

if 'expander_open' not in st.session_state: 
    st.session_state.expander_open = True

# Funções de Callback (Garantem que o botão funcione na Nuvem)
def unlock_tab1():
    st.session_state.access_key_tab1_vFinal = True

def unlock_tab3():
    st.session_state.access_key_tab3_vFinal = True

def close_expander(): 
    st.session_state.expander_open = False

# --- IMPORTS GERAIS ---
import pandas as pd
import plotly.graph_objects as go
import feedparser
import yfinance as yf
import datetime
from datetime import timedelta
from time import mktime
import json
import os
import uuid

# Tenta importar fundamentus
try:
    import fundamentus
except ImportError:
    st.error("Biblioteca 'fundamentus' não encontrada. Adicione ao requirements.txt")
    st.stop()

# --- CSS Global ---
st.markdown("""
    <style>
    [data-testid="stDataFrame"] table tr th { text-align: right !important; }
    [data-testid="stDataFrame"] table tr td { text-align: left !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #ff4b4b; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÃO DO CONTADOR DE VISITANTES ---
def update_visitor_counter():
    file_path = "visitor_counter.json"
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    try:
        if hasattr(st, "query_params"):
            current_params = st.query_params
            visitor_id = current_params.get("visitor_id", None)
        else:
            current_params = st.experimental_get_query_params()
            visitor_id = current_params.get("visitor_id", [None])[0]
    except:
        visitor_id = None

    if not visitor_id:
        visitor_id = str(uuid.uuid4())
        # Evita setar query params se não necessário para não recarregar a página na nuvem
        # if hasattr(st, "query_params"): st.query_params["visitor_id"] = visitor_id
        
    data = {"total_visits": 0, "daily_visits": {}}

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except: pass 

    if today not in data["daily_visits"]:
        data["daily_visits"][today] = []

    if visitor_id not in data["daily_visits"][today]:
        data["daily_visits"][today].append(visitor_id)
        data["total_visits"] += 1
        with open(file_path, "w") as f:
            json.dump(data, f)
            
    return data["total_visits"]

try:
    total_visitantes = update_visitor_counter()
except:
    total_visitantes = 0 

# --- BARRA LATERAL ---
with st.sidebar:
    # --- NOVO: BOTÃO WHATSAPP GIGA+ ---
    whatsapp_number = "552220410353"
    whatsapp_msg = "Use o codigo DVT329 e ganhe 20% nas duas primeiras mensalidades"
    whatsapp_url = f"https://wa.me/{whatsapp_number}?text={whatsapp_msg.replace(' ', '%20')}"
    
    st.markdown(f"""
        <a href="{whatsapp_url}" target="_blank" style="text-decoration: none;">
            <div style="
                background-color: #25D366; 
                padding: 12px; 
                border-radius: 10px; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                gap: 10px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.2); 
                margin-bottom: 20px;
                transition: transform 0.2s;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12.0117 2C6.50574 2 2.02344 6.47837 2.02344 11.9841C2.02344 13.7432 2.48045 15.4608 3.36889 16.9982L2 22L7.1264 20.6558C8.59864 21.4589 10.2798 21.8805 11.9839 21.8805H12.0089C17.5149 21.8805 21.9953 17.4022 21.9953 11.8965C21.9953 9.25595 20.9666 6.77209 19.098 4.90706C17.2295 3.04204 14.7438 2.01235 12.0117 2Z"/>
                </svg>
                <div style="color: white; font-weight: bold; font-size: 14px; line-height: 1.2;">
                    Ganhe 20% OFF<br>Giga+ Fibra!
                </div>
            </div>
        </a>
    """, unsafe_allow_html=True)
    # ----------------------------------

    st.header("📊 Estatísticas")
    st.metric(label="Visitantes Únicos", value=total_visitantes)
    st.divider()
    st.caption("Desenvolvido com Streamlit")

# --- Auxiliares ---
def clean_fundamentus_col(x):
    if pd.isna(x) or x == '': return 0.0
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            x = x.replace('%', '').replace('.', '').replace(',', '.')
            try: return float(x) / 100
            except: return 0.0
        x = x.replace('.', '').replace(',', '.')
        try: return float(x)
        except: return 0.0
    return 0.0

def format_short_number(val):
    if pd.isna(val) or val == 0: return ""
    abs_val = abs(val)
    if abs_val >= 1e9: return f"{val/1e9:.1f}B"
    elif abs_val >= 1e6: return f"{val/1e6:.0f}M"
    return f"{val:.0f}"

def get_current_data():
    now = datetime.datetime.now()
    return now.strftime("%B"), now.year

# --- Dados Principais ---
@st.cache_data(ttl=3600*6)
def get_ranking_data():
    try:
        df = fundamentus.get_resultado()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'papel'}, inplace=True)
        cols = ['pl', 'roe', 'dy', 'evebit', 'cotacao', 'liq2m', 'mrgliq', 'divbpatr', 'c5y']
        for col in cols:
            if col in df.columns: df[col] = df[col].apply(clean_fundamentus_col)
            else: df[col] = 0.0
        return df
    except: return pd.DataFrame()

# Filtros da Tabela "Melhores"
def apply_best_filters(df):
    if df.empty: return df
    filtro = (
        (df['roe'] > 0.05) & (df['pl'] < 15) & (df['pl'] > 0) & 
        (df['evebit'] > 0) & (df['evebit'] < 10) &
        (df['dy'] > 0.04) & (df['mrgliq'] > 0.05) & (df['liq2m'] > 200000)
    )
    df_filtered = df[filtro].copy()
    df_filtered['dy'] = df_filtered['dy'] * 100
    df_filtered['mrgliq'] = df_filtered['mrgliq'] * 100
    df_filtered['roe'] = df_filtered['roe'] * 100
    df_filtered.rename(columns={
        'papel': 'Ativo', 'cotacao': 'Preço', 'pl': 'P/L', 
        'evebit': 'EV/EBIT', 'dy': 'DY', 'roe': 'ROE', 'mrgliq': 'Margem Líq.'
    }, inplace=True)
    return df_filtered.sort_values(by=['P/L', 'Margem Líq.'], ascending=[True, False]).reset_index(drop=True)

# --- Tabela de Risco ---
@st.cache_data(ttl=3600*12)
def get_risk_table(df_original):
    if df_original.empty: return pd.DataFrame()
    lista_rj = ['OIBR3', 'OIBR4', 'AMER3', 'GOLL4', 'AZUL4', 'RCSL3', 'RCSL4', 'RSID3', 'TCNO3', 'TCNO4']
    mask_risk = (df_original['divbpatr'] > 3.0) | (df_original['papel'].isin(lista_rj))
    df_risk = df_original[mask_risk].copy()
    if df_risk.empty: return pd.DataFrame()

    risk_data = []
    top_risks = df_risk.sort_values(by='divbpatr', ascending=False).head(15)
    
    for idx, row in top_risks.iterrows():
        ticker = row['papel']
        status = "Recup. Judicial" if ticker in lista_rj else "Alta Alavancagem"
        lucro_queda_str = "N/D"
        val_queda = 0.0
        try:
            stock = yf.Ticker(ticker + ".SA")
            fin = stock.financials
            if not fin.empty:
                inc_row = None
                possible_names = ['Net Income', 'Net Income Common']
                for name in possible_names:
                    if name in fin.index:
                        inc_row = fin.loc[name]
                        break
                if inc_row is not None and len(inc_row) >= 2:
                    curr = inc_row.iloc[0]
                    prev = inc_row.iloc[1]
                    if curr < prev:
                        pct = ((curr - prev) / abs(prev)) * 100
                        val_queda = pct 
                        lucro_queda_str = f"{pct:.1f}%"
                    else: lucro_queda_str = "Estável"
        except: pass

        if val_queda < 0 or ticker in lista_rj:
            risk_data.append({
                'Ativo': ticker, 'Preço': row['cotacao'],
                'Alavancagem': row['divbpatr'], 'Queda Lucro': lucro_queda_str,
                'Situação': status
            })
    return pd.DataFrame(risk_data)

# --- Lógica do Gráfico ---
@st.cache_data(ttl=3600*24)
def get_chart_data(ticker):
    try:
        stock = yf.Ticker(ticker + ".SA")
        financials = stock.financials.T
        quarterly = stock.quarterly_financials.T
        hist = stock.history(period="5y")
        
        if not financials.empty: 
            financials.index = pd.to_datetime(financials.index).tz_localize(None)
            financials = financials.sort_index()
        if not quarterly.empty: 
            quarterly.index = pd.to_datetime(quarterly.index).tz_localize(None)
            quarterly = quarterly.sort_index()
        if not hist.empty: 
            hist.index = pd.to_datetime(hist.index).tz_localize(None)

        def find_col(df, candidates):
            cols = [c for c in df.columns]
            for cand in candidates:
                for col in cols:
                    if cand.lower() == col.lower() or cand.lower() in col.lower():
                        return col
            return None

        rev_candidates = ['Total Revenue', 'Operating Revenue', 'Revenue', 'Receita Total']
        inc_candidates = ['Net Income', 'Net Income Common', 'Net Income Continuous', 'Lucro Liquido']

        if financials.empty: return None

        rev_col = find_col(financials, rev_candidates)
        inc_col = find_col(financials, inc_candidates)
        
        if not rev_col or not inc_col: return None

        data_rows = []
        last_3_years = financials.tail(3)
        for date, row in last_3_years.iterrows():
            year_str = str(date.year)
            price = 0.0
            if not hist.empty:
                df_yr = hist[hist.index.year == date.year]
                if not df_yr.empty: price = df_yr['Close'].iloc[-1]
                else:
                    mask = hist.index <= date
                    if mask.any(): price = hist.loc[mask, 'Close'].iloc[-1]
            data_rows.append({'Periodo': year_str, 'Receita': row[rev_col], 'Lucro': row[inc_col], 'Cotação': price})
            
        ttm_rev, ttm_inc, has_ttm = 0, 0, False
        if not quarterly.empty:
            q_limit = min(4, len(quarterly))
            last_q = quarterly.tail(q_limit)
            q_rev_col = find_col(quarterly, rev_candidates)
            q_inc_col = find_col(quarterly, inc_candidates)
            if q_rev_col and q_inc_col:
                ttm_rev = last_q[q_rev_col].sum()
                ttm_inc = last_q[q_inc_col].sum()
                has_ttm = True
        
        if has_ttm:
            curr_price = 0.0
            if not hist.empty: curr_price = hist['Close'].iloc[-1]
            data_rows.append({'Periodo': 'Últimos 12m', 'Receita': ttm_rev, 'Lucro': ttm_inc, 'Cotação': curr_price})
        
        df_final = pd.DataFrame(data_rows)
        df_final['Receita_Texto'] = df_final['Receita'].apply(format_short_number)
        return df_final
    except: return None


# --- Dividendos ---
@st.cache_data(ttl=3600*6)
def get_latest_dividends(ticker_list):
    divs_data = []
    for ticker in ticker_list[:10]:
        try:
            stock = yf.Ticker(ticker + ".SA")
            d = stock.dividends
            if not d.empty: divs_data.append({'Ativo': ticker, 'Valor': d.iloc[-1], 'Data': d.index[-1]})
        except: continue
    if divs_data:
        df = pd.DataFrame(divs_data)
        df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
        return df.sort_values('Data', ascending=False).head(5)
    return pd.DataFrame()

# --- Notícias ---
@st.cache_data(ttl=1800)
def get_market_news():
    feeds = {'Money Times': 'https://www.moneytimes.com.br/feed/', 'InfoMoney': 'https://www.infomoney.com.br/feed/'}
    news_items = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                news_items.append({'title': entry.title, 'link': entry.link, 'source': source})
        except: continue
    return news_items[:6]

# --- SCANNER BOLLINGER (MÉDIA CENTRAL - SMA 20) ---
@st.cache_data(ttl=600)
def scan_bollinger_weekly_central(df_base):
    if df_base.empty: return pd.DataFrame()
    try:
        # Top 300 Liquidez
        top_300 = df_base.sort_values(by='liq2m', ascending=False).head(300)['papel'].tolist()
        tickers_br = [t + ".SA" for t in top_300]
    except: return pd.DataFrame()
    
    candidates = []
    try:
        data = yf.download(tickers_br, period="2y", interval="1wk", group_by='ticker', progress=False, threads=True)
        for t in tickers_br:
            try:
                df_t = data[t].copy() if t in data else pd.DataFrame()
                if df_t.empty: continue
                df_t.dropna(subset=['Close'], inplace=True)
                if len(df_t) < 22: continue 

                df_t['SMA20'] = df_t['Close'].rolling(window=20).mean()
                
                curr = df_t.iloc[-1]
                prev = df_t.iloc[-2]
                
                def check_pullback_sma(row):
                    # Tocou na média (Minima <= Média) E Fechou acima (Close > Média)
                    return (row['Low'] <= row['SMA20']) and (row['Close'] > row['SMA20'])

                found = False
                periodo = ""
                
                if check_pullback_sma(curr):
                    found, periodo = True, "Semana Atual"
                elif check_pullback_sma(prev):
                    found, periodo = True, "Semana Anterior"

                if found:
                    clean = t.replace(".SA", "")
                    dist = ((curr['Close'] - curr['SMA20']) / curr['SMA20']) * 100
                    candidates.append({
                        'Ativo': clean, 'Setup': periodo, 'Preço Atual': curr['Close'],
                        'Dist. Média %': dist, 'TV_Symbol': f"BMFBOVESPA:{clean}"
                    })
            except: continue
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- SCANNER ROC ---
@st.cache_data(ttl=3600*4)
def scan_roc_weekly(df_top_liq):
    if df_top_liq.empty: return pd.DataFrame()
    top_tickers = df_top_liq.sort_values(by='liq2m', ascending=False).head(90)['papel'].tolist()
    tickers_sa = [t + ".SA" for t in top_tickers]
    candidates = []
    try:
        data = yf.download(tickers_sa, period="7y", interval="1wk", group_by='ticker', progress=False, threads=True)
        for t in tickers_sa:
            clean = t.replace(".SA", "")
            try:
                df_t = data[t].copy() if t in data else pd.DataFrame()
                if df_t.empty or len(df_t) < 310: continue
                
                df_t['EMA17'] = df_t['Close'].ewm(span=17).mean()
                df_t['EMA34'] = df_t['Close'].ewm(span=34).mean()
                df_t['EMA72'] = df_t['Close'].ewm(span=72).mean()
                df_t['EMA305'] = df_t['Close'].ewm(span=305).mean()
                
                curr = df_t.iloc[-1]
                
                # ROC
                roc17 = ((curr['Close'] - curr['EMA17']) / curr['EMA17']) * 100
                roc34 = ((curr['Close'] - curr['EMA34']) / curr['EMA34']) * 100
                roc72 = ((curr['Close'] - curr['EMA72']) / curr['EMA72']) * 100
                roc305 = ((curr['Close'] - curr['EMA305']) / curr['EMA305']) * 100

                cond_alta = (roc17 < 0) & (roc34 > 0) & (roc72 > 0) & (roc305 > 0)
                cond_media = (roc17 < 0) & (roc34 < 0) & (roc34 > roc17) & (roc72 > 0) & (roc305 > 0)
                
                prob = "Alta (Caiu Comprou)" if cond_alta else ("Média" if cond_media else "")
                
                if prob:
                    candidates.append({
                        'Ativo': clean, 'Preço': curr['Close'], 'Probabilidade': prob,
                        'ROC17 %': roc17, 'TV_Symbol': f"BMFBOVESPA:{clean}"
                    })
            except: continue
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- Widget TV ---
def show_chart_widget(symbol_tv, interval="D"):
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%", "height": 500, "symbol": "{symbol_tv}", "interval": "{interval}", 
        "timezone": "America/Sao_Paulo", "theme": "light", "style": "1", "locale": "br",
        "toolbar_bg": "#f1f3f6", "enable_publishing": false, "allow_symbol_change": true,
        "studies": ["MASimple@tv-basicstudies"], "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(html_code, height=500)

# ==========================================
# INTERFACE
# ==========================================

# Banner Principal
components.html("""
    <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
        <script src="https://pl28325401.effectivegatecpm.com/1a/83/79/1a8379a4a8ddb94a327a5797257a9f02.js"></script>
    </div>
""", height=90)

st.title("🇧🇷 Ranking de Ações Baratas e Rentáveis - B3")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")
st.warning("⚠️ Utilize os dados apenas para estudo.")

# Carrega Dados
with st.spinner('Analisando mercado...'):
    df_raw = get_ranking_data()
    df_best = apply_best_filters(df_raw)
    df_warning = get_risk_table(df_raw)
    df_scan_bb = scan_bollinger_weekly_central(df_raw) 
    df_scan_roc = scan_roc_weekly(df_raw)

tab1, tab2, tab3 = st.tabs(["🏆 Ranking Fundamentalista", "📉 Setup Média 20 (Pullback)", "🚀 Setup ROC (Caiu Comprou)"])

# === ABA 1 (COM BLOQUEIO REFORÇADO) ===
with tab1:
    st.markdown("Oportunidades Fundamentalistas.")
    
    # Verifica Estado
    if not st.session_state.access_key_tab1_vFinal:
        st.info("🔒 **Lista Protegida:** Para visualizar as melhores ações, clique no banner abaixo.")
        
        components.html("""
            <div style="display: flex; justify-content: center; align-items: center; width: 100%; height: 100%; border: 1px dashed #ccc;">
                <script src="https://pl28325401.effectivegatecpm.com/1a/83/79/1a8379a4a8ddb94a327a5797257a9f02.js"></script>
            </div>
        """, height=130)
        
        # O uso de on_click garante o update antes do rerun
        st.button("🔓 Já cliquei no banner / Liberar Lista", on_click=unlock_tab1)
            
    else:
        # CONTEÚDO LIBERADO
        if not df_best.empty:
            st.subheader("🏆 Melhores Ações")
            cols_view = ['Ativo', 'Preço', 'EV/EBIT', 'P/L', 'ROE', 'DY', 'Margem Líq.']
            styler = df_best[cols_view].style.map(lambda x: 'background-color: #f2f2f2; color: black;', subset=['Preço','P/L']).format({"Preço": "R$ {:.2f}", "P/L": "{:.2f}", "DY": "{:.2f}"})
            st.dataframe(styler, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("⚠️ Empresas em Risco")
        if not df_warning.empty:
            styler_risk = df_warning.style.map(lambda v: 'color: red;' if '-' in str(v) else '', subset=['Queda Lucro']).format({"Preço": "R$ {:.2f}"})
            st.dataframe(styler_risk, use_container_width=True, hide_index=True)
        else: st.info("Sem alertas hoje.")

        st.divider()
        st.subheader("📈 Gráfico")
        opts = df_best['Ativo'].tolist()
        idx = opts.index('LREN3') if 'LREN3' in opts else 0
        with st.expander("🔎 Selecionar Ação", expanded=st.session_state.expander_open):
            selected = st.selectbox("Ativo:", opts, index=idx, on_change=close_expander)
        if selected:
            df_chart = get_chart_data(selected)
            if df_chart is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_chart['Periodo'], y=df_chart['Receita'], name="Receita", marker_color='#A9A9A9'))
                fig.add_trace(go.Scatter(x=df_chart['Periodo'], y=df_chart['Lucro'], name="Lucro", line=dict(color='green', width=3)))
                fig.add_trace(go.Scatter(x=df_chart['Periodo'], y=df_chart['Cotação'], name="Preço", line=dict(color='blue', width=3), yaxis='y3'))
                fig.update_layout(title=f"{selected}", yaxis3=dict(overlaying="y", side="right"), height=400)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📰 Notícias")
            for n in get_market_news(): st.markdown(f"[{n['title']}]({n['link']})")
        with c2:
            st.subheader("💰 Dividendos")
            df_d = get_latest_dividends(df_best['Ativo'].tolist())
            if not df_d.empty: 
                df_d['Valor'] = df_d['Valor'].apply(lambda x: f"R$ {x:.4f}")
                df_d['Data'] = df_d['Data'].dt.strftime('%d/%m/%Y')
                st.dataframe(df_d, hide_index=True)

# === ABA 2 (SETUP PULLBACK MEDIA) ===
with tab2:
    st.subheader("📉 Setup: Pullback na Média de 20 (Semanal)")
    st.markdown("**Critérios:** Top 300 Liquidez | Mínima tocou Média 20 | Fechou Acima da Média.")
    
    col_list, col_chart = st.columns([1, 2])
    with col_list:
        if not df_scan_bb.empty:
            st.write(f"**{len(df_scan_bb)} Ativos:**")
            event = st.dataframe(df_scan_bb[['Ativo', 'Setup', 'Preço Atual', 'Dist. Média %']].style.format({"Preço Atual": "{:.2f}", "Dist. Média %": "{:.2f}%"}), use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
            if len(event.selection.rows) > 0:
                st.session_state.tv_symbol = df_scan_bb.iloc[event.selection.rows[0]]['TV_Symbol']
        else: st.info("Sem sinais.")
    with col_chart:
        clean = st.session_state.tv_symbol.split(":")[-1]
        st.markdown(f"#### {clean} (Semanal)")
        show_chart_widget(st.session_state.tv_symbol, interval="W")

# === ABA 3 (COM BLOQUEIO REFORÇADO) ===
with tab3:
    if not st.session_state.access_key_tab3_vFinal:
        st.warning("🔒 Conteúdo Bloqueado")
        st.info("Para liberar o Setup ROC, clique no banner abaixo.")
        
        components.html("""
            <div style="display: flex; justify-content: center; align-items: center; width: 100%; height: 100%; border: 1px dashed #ccc;">
                <script src="https://pl28325401.effectivegatecpm.com/1a/83/79/1a8379a4a8ddb94a327a5797257a9f02.js"></script>
            </div>
        """, height=130)
        
        st.button("🔓 Já cliquei no banner / Liberar Acesso", on_click=unlock_tab3)
            
    else:
        st.success("Liberado!")
        st.subheader("🚀 Setup ROC")
        col_roc_list, col_roc_chart = st.columns([1, 2])
        with col_roc_list:
            if not df_scan_roc.empty:
                if st.session_state.tv_symbol == "BMFBOVESPA:LREN3": 
                    st.session_state.tv_symbol = df_scan_roc.iloc[0]['TV_Symbol']
                def color_prob(val): return f'background-color: {"#d4edda" if "Alta" in val else "#fff3cd"}; color: black;'
                event_roc = st.dataframe(df_scan_roc[['Ativo', 'Preço', 'Probabilidade', 'ROC17 %']].style.format({"Preço": "R$ {:.2f}", "ROC17 %": "{:.2f}%"}).map(color_prob, subset=['Probabilidade']), use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="roc_table")
                if len(event_roc.selection.rows) > 0: st.session_state.tv_symbol = df_scan_roc.iloc[event_roc.selection.rows[0]]['TV_Symbol']
            else: st.info("Sem sinais ROC.")
        with col_roc_chart:
            clean = st.session_state.tv_symbol.split(":")[-1]
            st.markdown(f"#### {clean} (Diário)")
            show_chart_widget(st.session_state.tv_symbol, interval="D")
