import streamlit as st
import streamlit.components.v1 as components

# --- 1. CONFIGURAÇÃO DA PÁGINA (PRIMEIRA LINHA OBRIGATÓRIA) ---
st.set_page_config(
    page_title="Melhores Ações Ibovespa 2025 | Ranking Fundamentalista",
    layout="wide",
    page_icon="🇧🇷"
)

# --- IMPORTS ---
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

# Tratamento de erro importação
try:
    import fundamentus
except ImportError:
    st.error("Biblioteca 'fundamentus' não encontrada. Adicione ao requirements.txt")
    st.stop()

# --- CSS GLOBAL ---
st.markdown("""
    <style>
    [data-testid="stDataFrame"] table tr th { text-align: right !important; }
    [data-testid="stDataFrame"] table tr td { text-align: left !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #ff4b4b; }
    div.stButton > button:first-child { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- CONTADOR DE VISITANTES ---
def update_visitor_counter():
    file_path = "visitor_counter.json"
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        if hasattr(st, "query_params"):
            visitor_id = st.query_params.get("visitor_id", None)
        else:
            visitor_id = st.experimental_get_query_params().get("visitor_id", [None])[0]
    except: visitor_id = None

    if not visitor_id:
        visitor_id = str(uuid.uuid4())
        if hasattr(st, "query_params"):
            try: st.query_params["visitor_id"] = visitor_id
            except: pass
        
    data = {"total_visits": 0, "daily_visits": {}}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f: data = json.load(f)
        except: pass 

    if today not in data["daily_visits"]: data["daily_visits"][today] = []
    if visitor_id not in data["daily_visits"][today]:
        data["daily_visits"][today].append(visitor_id)
        data["total_visits"] += 1
        with open(file_path, "w") as f: json.dump(data, f)
    return data["total_visits"]

try: total_visitantes = update_visitor_counter()
except: total_visitantes = 0 

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Estatísticas")
    st.metric(label="Visitantes Únicos", value=total_visitantes)
    st.divider()
    st.caption("Desenvolvido com Streamlit")

# --- ESTADO ---
if 'expander_open' not in st.session_state: st.session_state.expander_open = True
if 'tv_symbol' not in st.session_state: st.session_state.tv_symbol = "BMFBOVESPA:LREN3"
def close_expander(): st.session_state.expander_open = False

# --- FUNÇÕES AUXILIARES ---
def clean_fundamentus_col(x):
    if pd.isna(x) or x == '': return 0.0
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            try: return float(x.replace('%', '').replace('.', '').replace(',', '.')) / 100
            except: return 0.0
        try: return float(x.replace('.', '').replace(',', '.'))
        except: return 0.0
    return 0.0

def format_short_number(val):
    if pd.isna(val) or val == 0: return ""
    if abs(val) >= 1e9: return f"{val/1e9:.1f}B"
    elif abs(val) >= 1e6: return f"{val/1e6:.0f}M"
    return f"{val:.0f}"

def get_current_data():
    now = datetime.datetime.now()
    return now.strftime("%B"), now.year

# --- DADOS FUNDAMENTUS (ABA 1) ---
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
                for name in ['Net Income', 'Net Income Common', 'Lucro Liquido']:
                    if name in fin.index:
                        curr = fin.loc[name].iloc[0]
                        prev = fin.loc[name].iloc[1]
                        if curr < prev:
                            pct = ((curr - prev) / abs(prev)) * 100
                            val_queda = pct 
                            lucro_queda_str = f"{pct:.1f}%"
                        else: lucro_queda_str = "Estável"
                        break
        except: pass
        if val_queda < 0 or ticker in lista_rj:
            risk_data.append({'Ativo': ticker, 'Preço': row['cotacao'], 'Alavancagem': row['divbpatr'], 'Queda Lucro': lucro_queda_str, 'Situação': status})
    return pd.DataFrame(risk_data)

# --- DADOS GRÁFICO (ABA 1) ---
@st.cache_data(ttl=3600*24)
def get_chart_data(ticker):
    try:
        stock = yf.Ticker(ticker + ".SA")
        financials = stock.financials.T.sort_index()
        quarterly = stock.quarterly_financials.T.sort_index()
        hist = stock.history(period="5y")
        if hist.empty or financials.empty: return None
        
        financials.index = pd.to_datetime(financials.index).tz_localize(None)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        
        # Encontrar colunas
        def find_col(df, terms):
            for t in terms:
                for c in df.columns:
                    if t.lower() in c.lower(): return c
            return None

        rev_col = find_col(financials, ['Total Revenue', 'Revenue', 'Receita'])
        inc_col = find_col(financials, ['Net Income', 'Lucro'])
        if not rev_col or not inc_col: return None

        data_rows = []
        for date, row in financials.tail(3).iterrows():
            price = 0.0
            mask = hist.index <= date
            if mask.any(): price = hist.loc[mask, 'Close'].iloc[-1]
            data_rows.append({'Periodo': str(date.year), 'Receita': row[rev_col], 'Lucro': row[inc_col], 'Cotação': price})
            
        # TTM
        if not quarterly.empty:
            last_q = quarterly.tail(4)
            rev_q = find_col(quarterly, ['Total Revenue', 'Revenue', 'Receita'])
            inc_q = find_col(quarterly, ['Net Income', 'Lucro'])
            if rev_q and inc_q:
                curr_price = hist['Close'].iloc[-1]
                data_rows.append({'Periodo': '12m (TTM)', 'Receita': last_q[rev_q].sum(), 'Lucro': last_q[inc_q].sum(), 'Cotação': curr_price})
        
        df = pd.DataFrame(data_rows)
        df['Receita_Texto'] = df['Receita'].apply(format_short_number)
        return df
    except: return None

# --- DIVIDENDOS & NOTICIAS (ABA 1) ---
@st.cache_data(ttl=3600*6)
def get_latest_dividends(ticker_list):
    data = []
    for t in ticker_list[:10]:
        try:
            d = yf.Ticker(t + ".SA").dividends
            if not d.empty: data.append({'Ativo': t, 'Valor': d.iloc[-1], 'Data': d.index[-1]})
        except: continue
    if data:
        df = pd.DataFrame(data)
        df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
        return df.sort_values('Data', ascending=False).head(5)
    return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_market_news():
    feeds = {'Money Times': 'https://www.moneytimes.com.br/feed/', 'InfoMoney': 'https://www.infomoney.com.br/feed/'}
    items = []
    for src, url in feeds.items():
        try:
            d = feedparser.parse(url)
            for e in d.entries[:2]:
                items.append({'title': e.title, 'link': e.link, 'source': src})
        except: continue
    return items

# --- SCANNER BOLLINGER CORRIGIDO (ABA 2) ---
@st.cache_data(ttl=600)
def scan_bollinger_weekly():
    # Lista de Ações Líquidas
    tickers_br = [
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "WEGE3.SA", "PRIO3.SA", 
        "MGLU3.SA", "HAPV3.SA", "RDOR3.SA", "SUZB3.SA", "JBSS3.SA", "RAIZ4.SA", "GGBR4.SA", "CSAN3.SA",
        "VBBR3.SA", "B3SA3.SA", "BBSE3.SA", "CMIG4.SA", "ITSA4.SA", "BHIA3.SA", "GOLL4.SA", "AZUL4.SA", 
        "CVCB3.SA", "USIM5.SA", "CSNA3.SA", "EMBR3.SA", "CPLE6.SA", "RADL3.SA", "EQTL3.SA", "TOTS3.SA", 
        "RENT3.SA", "TIMS3.SA", "SBSP3.SA", "ELET3.SA", "ABEV3.SA", "ASAI3.SA", "CRFB3.SA", "MULT3.SA",
        "CYRE3.SA", "EZTC3.SA", "MRVE3.SA", "PETZ3.SA", "SOMA3.SA", "ALPA4.SA", "LREN3.SA", "SMTO3.SA",
        "UGPA3.SA", "ENEV3.SA", "EGIE3.SA", "CPFE3.SA", "HYPE3.SA", "LIGT3.SA", "YDUQ3.SA", "COGN3.SA"
    ]
    
    candidates = []
    
    try:
        # IMPORTANTE: auto_adjust=False para alinhar com preço nominal (TradingView)
        data = yf.download(tickers_br, period="2y", interval="1wk", group_by='ticker', auto_adjust=False, progress=False, threads=True)
        
        for t in tickers_br:
            try:
                # Extrai dados do ticker
                if len(tickers_br) > 1:
                    df_t = data[t].copy() if t in data else pd.DataFrame()
                else:
                    df_t = data.copy()
                
                # Limpeza rigorosa
                if df_t.empty: continue
                
                # Garante que temos colunas essenciais
                if 'Close' not in df_t.columns or 'Low' not in df_t.columns: continue

                # Remove linhas sem Close (feriados ou erros)
                df_t.dropna(subset=['Close', 'Low'], inplace=True)
                
                # Precisa de histórico para média (20 periodos)
                if len(df_t) < 22: continue 

                # Cálculo Bandas Bollinger (20, 2)
                df_t['SMA20'] = df_t['Close'].rolling(window=20).mean()
                df_t['STD20'] = df_t['Close'].rolling(window=20).std()
                df_t['Lower'] = df_t['SMA20'] - (2.0 * df_t['STD20'])
                
                # Remove o início onde SMA é NaN
                df_t.dropna(subset=['Lower'], inplace=True)
                
                if df_t.empty: continue

                # Pega a vela MAIS RECENTE disponível
                curr = df_t.iloc[-1]
                
                # === LÓGICA CORRIGIDA E ESTRITA ===
                # Seleciona APENAS se a MÍNIMA (Low) for MENOR ou IGUAL à Banda Inferior
                # Isso cobre toques (pavio) e rompimentos (fechamento abaixo)
                if curr['Low'] <= curr['Lower']:
                    clean_ticker = t.replace(".SA", "")
                    
                    status_txt = "Furou (Fech < Banda)" if curr['Close'] < curr['Lower'] else "Tocou (Pavio)"
                    
                    candidates.append({
                        'Ativo': clean_ticker,
                        'Preço': curr['Close'],
                        'Mínima Sem.': curr['Low'],
                        'Banda Inf': curr['Lower'],
                        'Status': status_txt,
                        'TV_Symbol': f"BMFBOVESPA:{clean_ticker}"
                    })

            except: continue
        
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- SCANNER ROC EMA (ABA 3) ---
@st.cache_data(ttl=3600*4)
def scan_roc_weekly(df_top_liq):
    if df_top_liq.empty: return pd.DataFrame()
    top_tickers = df_top_liq.sort_values(by='liq2m', ascending=False).head(90)['papel'].tolist()
    tickers_sa = [t + ".SA" for t in top_tickers]
    candidates = []
    try:
        data = yf.download(tickers_sa, period="7y", interval="1wk", group_by='ticker', progress=False, threads=True)
        for t in tickers_sa:
            clean_ticker = t.replace(".SA", "")
            try:
                df_t = data[t].copy() if t in data else pd.DataFrame()
                if df_t.empty: continue
                df_t.dropna(subset=['Close'], inplace=True)
                if len(df_t) < 310: continue
                
                df_t['EMA17'] = df_t['Close'].ewm(span=17).mean()
                df_t['EMA34'] = df_t['Close'].ewm(span=34).mean()
                df_t['EMA72'] = df_t['Close'].ewm(span=72).mean()
                df_t['EMA305'] = df_t['Close'].ewm(span=305).mean()
                
                curr = df_t.iloc[-1]
                
                cond_alta = (curr['Close'] < curr['EMA17']) and (curr['Close'] > curr['EMA34']) and (curr['Close'] > curr['EMA72']) and (curr['Close'] > curr['EMA305'])
                cond_media = (curr['Close'] < curr['EMA34']) and (curr['EMA34'] > curr['EMA72']) and (curr['EMA72'] > curr['EMA305'])
                
                prob = ""
                if cond_alta: prob = "Alta (Caiu Comprou)"
                elif cond_media: prob = "Média (Pullback Profundo)"
                
                if prob:
                    roc17 = ((curr['Close'] - curr['EMA17']) / curr['EMA17']) * 100
                    candidates.append({'Ativo': clean_ticker, 'Preço': curr['Close'], 'Probabilidade': prob, 'ROC17 %': roc17, 'TV_Symbol': f"BMFBOVESPA:{clean_ticker}"})
            except: continue
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- WIDGET TV ---
def show_chart_widget(symbol_tv, interval="D"):
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%", "height": 500,
        "symbol": "{symbol_tv}",
        "interval": "{interval}", 
        "timezone": "America/Sao_Paulo",
        "theme": "light", "style": "1", "locale": "br", "toolbar_bg": "#f1f3f6", "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": ["MASimple@tv-basicstudies", "BB@tv-basicstudies"], 
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(html_code, height=500)

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

components.html("""<div style="display:flex;justify-content:center;width:100%;"><script src="https://pl28325401.effectivegatecpm.com/1a/83/79/1a8379a4a8ddb94a327a5797257a9f02.js"></script></div>""", height=90)

st.title("🇧🇷 Ranking e Scanners - B3")
st.warning("⚠️ **AVISO:** Utilize os dados apenas para estudo.")

# CARREGAMENTO
with st.spinner('Analisando mercado...'):
    df_raw = get_ranking_data()
    df_best = apply_best_filters(df_raw)
    df_warning = get_risk_table(df_raw)
    df_scan_bb = scan_bollinger_weekly()
    df_scan_roc = scan_roc_weekly(df_raw)

tab1, tab2, tab3 = st.tabs(["🏆 Fundamentalista", "📉 Setup Bollinger (Semanal)", "🚀 Setup ROC (Caiu Comprou)"])

# === ABA 1 ===
with tab1:
    st.markdown("Filtro automático de boas empresas.")
    if not df_best.empty:
        st.dataframe(df_best[['Ativo', 'Preço', 'P/L', 'ROE', 'DY', 'Margem Líq.']].style.format({"Preço": "R$ {:.2f}", "P/L": "{:.2f}", "ROE": "{:.2f}", "DY": "{:.2f}", "Margem Líq.": "{:.2f}"}), use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("⚠️ Empresas em Risco / Recup. Judicial")
    if not df_warning.empty:
        st.dataframe(df_warning.style.format({"Preço": "R$ {:.2f}", "Alavancagem": "{:.2f}"}), use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("🔎 Gráfico Rápido")
    opt = df_best['Ativo'].tolist()
    sel = st.selectbox("Ativo:", opt, index=0 if opt else None)
    if sel:
        d = get_chart_data(sel)
        if d is not None:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=d['Periodo'], y=d['Receita'], name="Receita", marker_color='silver'))
            fig.add_trace(go.Scatter(x=d['Periodo'], y=d['Lucro'], name="Lucro", line=dict(color='green', width=3)))
            fig.add_trace(go.Scatter(x=d['Periodo'], y=d['Cotação'], name="Cotação", line=dict(color='blue'), yaxis='y2'))
            fig.update_layout(title=sel, yaxis2=dict(overlaying='y', side='right'), height=400)
            st.plotly_chart(fig, use_container_width=True)

# === ABA 2 (CORRIGIDA) ===
with tab2:
    st.subheader("📉 Bandas de Bollinger Semanal (Mínima tocou Banda Inf.)")
    st.info("Mostra apenas ações onde a **Mínima (Low)** da semana atual/recente é **menor ou igual** à Banda Inferior.")
    
    col_list, col_chart = st.columns([1, 2])
    with col_list:
        if not df_scan_bb.empty:
            st.write(f"**{len(df_scan_bb)} Oportunidades:**")
            
            def color_st(v): return f"background-color: {'#ffcccc' if 'Furou' in v else '#fffacd'}; color: black"
            
            evt = st.dataframe(
                df_scan_bb[['Ativo', 'Preço', 'Mínima Sem.', 'Banda Inf', 'Status']].style.format({
                    "Preço": "{:.2f}", "Mínima Sem.": "{:.2f}", "Banda Inf": "{:.2f}"
                }).map(color_st, subset=['Status']),
                use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
            )
            if len(evt.selection.rows) > 0:
                st.session_state.tv_symbol = df_scan_bb.iloc[evt.selection.rows[0]]['TV_Symbol']
        else:
            st.info("Nenhuma ação tocando a banda inferior nesta semana.")
            
    with col_chart:
        nm = st.session_state.tv_symbol.split(":")[-1]
        st.markdown(f"#### {nm} (Semanal)")
        show_chart_widget(st.session_state.tv_symbol, interval="W")

# === ABA 3 ===
with tab3:
    st.subheader("🚀 Setup ROC (Recuo na Tendência)")
    c1, c2 = st.columns([1, 2])
    with c1:
        if not df_scan_roc.empty:
            evt_roc = st.dataframe(
                df_scan_roc[['Ativo', 'Preço', 'Probabilidade']].style.format({"Preço": "{:.2f}"}),
                use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
            )
            if len(evt_roc.selection.rows) > 0:
                st.session_state.tv_symbol = df_scan_roc.iloc[evt_roc.selection.rows[0]]['TV_Symbol']
        else: st.info("Sem ativos no setup ROC hoje.")
    with c2:
        nm = st.session_state.tv_symbol.split(":")[-1]
        st.markdown(f"#### {nm} (Diário)")
        show_chart_widget(st.session_state.tv_symbol, interval="D")
