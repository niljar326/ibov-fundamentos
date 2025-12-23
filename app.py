import streamlit as st
import streamlit.components.v1 as components # Necessário para o TradingView

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Melhores Ações Ibovespa 2025 | Ranking & Setup BB",
    layout="wide",
    page_icon="🇧🇷"
)

# --- IMPORTS GERAIS ---
import pandas as pd
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
    /* Cabeçalhos à direita */
    [data-testid="stDataFrame"] table tr th { text-align: right !important; }
    /* Células à esquerda */
    [data-testid="stDataFrame"] table tr td { text-align: left !important; }
    /* Ajuste visual das abas */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #ff4b4b; }
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
    except: visitor_id = None

    if not visitor_id:
        visitor_id = str(uuid.uuid4())
        if hasattr(st, "query_params"): st.query_params["visitor_id"] = visitor_id
        
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

with st.sidebar:
    st.header("📊 Estatísticas")
    st.metric(label="Visitantes Únicos", value=total_visitantes)
    st.divider()
    st.caption("Desenvolvido com Streamlit")

# --- Estado ---
if 'expander_open' not in st.session_state: st.session_state.expander_open = True
def close_expander(): st.session_state.expander_open = False

# --- Auxiliares ---
def clean_fundamentus_col(x):
    if pd.isna(x) or x == '': return 0.0
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        x = x.strip().replace('%', '').replace('.', '').replace(',', '.')
        try: return float(x) / 100 if x.endswith('%') else float(x)
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

# --- DADOS FUNDAMENTALISTAS ---
@st.cache_data(ttl=3600*6)
def get_ranking_data():
    try:
        df = fundamentus.get_resultado()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'papel'}, inplace=True)
        cols = ['pl', 'roe', 'dy', 'evebit', 'cotacao', 'liq2m', 'mrgliq', 'divbpatr']
        for col in cols:
            if col in df.columns: df[col] = df[col].apply(clean_fundamentus_col)
            else: df[col] = 0.0
        return df
    except: return pd.DataFrame()

def apply_best_filters(df):
    if df.empty: return df
    filtro = ((df['roe'] > 0.05) & (df['pl'] < 15) & (df['pl'] > 0) & (df['dy'] > 0.04) & (df['liq2m'] > 200000))
    df_filtered = df[filtro].copy()
    df_filtered[['dy', 'mrgliq', 'roe']] = df_filtered[['dy', 'mrgliq', 'roe']] * 100
    df_filtered.rename(columns={'papel': 'Ativo', 'cotacao': 'Preço', 'pl': 'P/L', 'evebit': 'EV/EBIT', 'dy': 'DY', 'roe': 'ROE', 'mrgliq': 'Margem Líq.'}, inplace=True)
    return df_filtered.sort_values(by=['P/L', 'Margem Líq.'], ascending=[True, False]).reset_index(drop=True)

# --- SCANNER BOLLINGER (BRASIL + EUA) ---
@st.cache_data(ttl=900)
def scan_bollinger_bands():
    # 1. Definir Listas de Ativos
    tickers_br = [
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "WEGE3.SA", "PRIO3.SA", "MGLU3.SA",
        "LREN3.SA", "HAPV3.SA", "RDOR3.SA", "SUZB3.SA", "JBSS3.SA", "RAIZ4.SA", "GGBR4.SA", "CSAN3.SA",
        "VBBR3.SA", "ELET3.SA", "B3SA3.SA", "BBSE3.SA", "CMIG4.SA", "ITSA4.SA", "VIIA3.SA", "GOLL4.SA",
        "AZUL4.SA", "CVCB3.SA", "USIM5.SA", "CSNA3.SA", "EMBR3.SA", "CPLE6.SA", "RADL3.SA", "EQTL3.SA"
    ]
    
    # Principais ações dos EUA (Sem sufixo .SA para lógica, mas precisamos saber a origem)
    tickers_us = [
        "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC", 
        "DIS", "KO", "PEP", "JPM", "V", "WMT", "PG", "XOM", "CVX", "BA"
    ]
    
    all_tickers = tickers_br + tickers_us
    
    candidates = []
    
    try:
        # Baixa dados (Intervalo Diário '1d' ou Semanal '1wk' conforme sua preferência, vou colocar diário para ser mais dinâmico)
        data = yf.download(all_tickers, period="6mo", interval="1d", group_by='ticker', progress=False, threads=True)
        
        for t in all_tickers:
            try:
                # Ajuste para pegar o DF correto
                df_t = data[t].copy() if t in data else pd.DataFrame()
                
                if df_t.empty: continue
                df_t.dropna(subset=['Close'], inplace=True)
                if len(df_t) < 20: continue

                # === CÁLCULO DAS BANDAS (Replicando o Pine Script) ===
                # length = 20, mult = 2.0
                df_t['SMA20'] = df_t['Close'].rolling(window=20).mean()
                df_t['STD20'] = df_t['Close'].rolling(window=20).std()
                df_t['Lower'] = df_t['SMA20'] - (2.0 * df_t['STD20'])
                
                # Pega a última vela
                curr = df_t.iloc[-1]
                
                # === LÓGICA DE RASTREIO ===
                # Price "Touching" Lower: Low <= Lower Band
                # Usamos tolerância de 1% (1.01) para pegar toques muito próximos
                if curr['Low'] <= (curr['Lower'] * 1.01):
                    
                    dist = ((curr['Close'] - curr['Lower']) / curr['Lower']) * 100
                    
                    # Identificar mercado para o Widget do TV
                    market_prefix = "BMFBOVESPA" if ".SA" in t else "NASDAQ"
                    clean_ticker = t.replace(".SA", "")
                    
                    # Correção para algumas US que podem ser NYSE
                    if t in ["DIS", "KO", "PEP", "JPM", "V", "WMT", "PG", "XOM", "CVX", "BA"]:
                        market_prefix = "NYSE"

                    candidates.append({
                        'Ativo': clean_ticker,
                        'Mercado': '🇧🇷 Brasil' if ".SA" in t else '🇺🇸 EUA',
                        'Preço': curr['Close'],
                        'Banda Inf': curr['Lower'],
                        'Distância %': dist,
                        'TV_Symbol': f"{market_prefix}:{clean_ticker}"
                    })
            except: continue
            
        return pd.DataFrame(candidates).sort_values('Distância %')
    except:
        return pd.DataFrame()

# --- FUNÇÃO DE NOTÍCIAS ---
@st.cache_data(ttl=1800)
def get_market_news():
    feeds = {
        'Money Times': 'https://www.moneytimes.com.br/feed/',
        'InfoMoney': 'https://www.infomoney.com.br/feed/',
        'E-Investidor': 'https://einvestidor.estadao.com.br/feed/'
    }
    news_items = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                try: dt_obj = datetime.datetime.fromtimestamp(mktime(entry.published_parsed)) - timedelta(hours=3)
                except: dt_obj = datetime.datetime.now()
                news_items.append({'title': entry.title, 'link': entry.link, 'date_obj': dt_obj, 'source': source})
        except: continue
    news_items.sort(key=lambda x: x['date_obj'], reverse=True)
    return news_items[:6]

# --- FUNÇÃO DE DIVIDENDOS ---
@st.cache_data(ttl=3600*6)
def get_latest_dividends(ticker_list):
    divs_data = []
    for ticker in ticker_list[:10]:
        try:
            stock = yf.Ticker(ticker + ".SA")
            d = stock.dividends
            if not d.empty:
                divs_data.append({'Ativo': ticker, 'Valor': d.iloc[-1], 'Data': d.index[-1]})
        except: continue
    if divs_data:
        df = pd.DataFrame(divs_data)
        df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
        return df.sort_values('Data', ascending=False).head(5)
    return pd.DataFrame()

# --- TABELA DE RISCO ---
@st.cache_data(ttl=3600*12)
def get_risk_table(df_original):
    if df_original.empty: return pd.DataFrame()
    lista_rj = ['OIBR3', 'OIBR4', 'AMER3', 'GOLL4', 'AZUL4', 'RCSL3', 'RCSL4']
    mask = (df_original['divbpatr'] > 3.0) | (df_original['papel'].isin(lista_rj))
    df_risk = df_original[mask].copy()
    if df_risk.empty: return pd.DataFrame()
    return df_risk[['papel', 'cotacao', 'divbpatr']].rename(columns={'papel':'Ativo', 'cotacao':'Preço', 'divbpatr':'Dív/Patr'}).head(10)

# --- WIDGET TRADINGVIEW CHART (DINÂMICO) ---
def show_chart_widget(symbol_tv):
    # Aqui injetamos o indicador BB visualmente
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 500,
        "symbol": "{symbol_tv}",
        "interval": "D",
        "timezone": "America/Sao_Paulo",
        "theme": "light",
        "style": "1",
        "locale": "br",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
          "BB@tv-basicstudies"
        ],
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(html_code, height=500)

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================
st.title("🇧🇷 Ranking B3 + Setup BB (Global)")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")

# 1. Carregamento dos Dados
with st.spinner('Processando Mercado (Ranking Fundamentalista + Scan Técnico)...'):
    df_raw = get_ranking_data()
    df_best = apply_best_filters(df_raw)
    df_warning = get_risk_table(df_raw)
    df_scan_bb = scan_bollinger_bands() # <--- SCANNER NOVO

# --- SISTEMA DE ABAS ---
tab1, tab2, tab3 = st.tabs(["🏆 Ranking Fundamentalista", "🌍 Rastreador Geral", "📉 Setup BB (Brasil & EUA)"])

# === ABA 1: CONTEÚDO ORIGINAL ===
with tab1:
    if not df_best.empty:
        st.subheader("🏆 Melhores Ações (Oportunidades)")
        st.caption("Filtro: P/L Baixo, Alta Rentabilidade e Dividendos.")
        
        cols_view = ['Ativo', 'Preço', 'EV/EBIT', 'P/L', 'ROE', 'DY', 'Margem Líq.']
        even_cols_subset = ['Preço', 'P/L', 'DY']
        styler = df_best[cols_view].style.map(
            lambda x: 'background-color: #f2f2f2; color: black;', 
            subset=even_cols_subset
        ).format({
            "Preço": "R$ {:.2f}", "EV/EBIT": "{:.2f}", "P/L": "{:.2f}",
            "ROE": "{:.2f}", "DY": "{:.2f}", "Margem Líq.": "{:.2f}"
        })
        st.dataframe(styler, use_container_width=True, hide_index=True)

    # --- BANNERS LADO A LADO ---
    st.divider()
    col_ad1, col_ad2 = st.columns(2)
    with col_ad1:
        st.info("✈️ **Nomad:** Taxa Zero em Dólar. Código: **Y39FP3XF8I**")
    with col_ad2:
        st.info("🤝 **Mercado Pago:** R$ 30 OFF no 1º uso. [Link](https://mpago.li/1VydVhw)")
    st.divider()

    # Tabela Risco
    st.subheader("⚠️ Atenção! Empresas em Risco")
    if not df_warning.empty:
        st.dataframe(df_warning.style.format({"Preço": "R$ {:.2f}", "Dív/Patr": "{:.2f}"}), hide_index=True)

    st.divider()

    # Notícias e Dividendos
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📰 Notícias")
        news = get_market_news()
        if news:
            for n in news: st.markdown(f"**[{n['title']}]({n['link']})**")
    with c2:
        st.subheader("💰 Dividendos")
        df_divs = get_latest_dividends(df_best['Ativo'].tolist() if not df_best.empty else [])
        if not df_divs.empty:
            df_divs['Data'] = df_divs['Data'].dt.strftime('%d/%m/%Y')
            st.dataframe(df_divs.style.format({"Valor": "R$ {:.4f}"}), hide_index=True)

# === ABA 2: RASTREADOR GERAL (WIDGET PADRÃO) ===
with tab2:
    st.subheader("Rastreador de Mercado (Tempo Real)")
    components.html("""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
      { "width": "100%", "height": 600, "defaultColumn": "overview", "defaultScreen": "general", "market": "brazil", "showToolbar": true, "colorTheme": "light", "locale": "br" }
      </script>
    </div>
    """, height=600)

# === ABA 3: SETUP BB (BRASIL + EUA) ===
with tab3:
    st.subheader("📉 Ações Tocando a Banda Inferior (B3 & EUA)")
    st.markdown("""
    Lista rastreada automaticamente de ativos onde a **Mínima do Dia** tocou ou furou a **Banda de Bollinger Inferior (20, 2)**.
    """)
    
    col_list, col_chart = st.columns([1, 2])
    
    selected_tv_symbol = "BMFBOVESPA:PETR4"
    
    with col_list:
        if not df_scan_bb.empty:
            st.write(f"**{len(df_scan_bb)} Oportunidades Encontradas:**")
            
            def color_dist(val):
                color = '#ffcccb' if val < 0 else '#e6fffa'
                return f'background-color: {color}; color: black'

            st.dataframe(
                df_scan_bb[['Ativo', 'Mercado', 'Preço', 'Banda Inf', 'Distância %']].style.format({
                    "Preço": "{:.2f}", "Banda Inf": "{:.2f}", "Distância %": "{:.2f}%"
                }).map(color_dist, subset=['Distância %']),
                use_container_width=True,
                hide_index=True
            )
            
            # Seletor
            sel_ticker = st.selectbox("Selecione para ver Gráfico:", df_scan_bb['Ativo'].tolist())
            
            # Pega o símbolo correto para o TV
            if sel_ticker:
                selected_tv_symbol = df_scan_bb.loc[df_scan_bb['Ativo'] == sel_ticker, 'TV_Symbol'].values[0]
                
        else:
            st.info("Nenhuma ação tocando a banda inferior hoje.")
            
    with col_chart:
        st.markdown(f"#### Análise: {selected_tv_symbol}")
        show_chart_widget(selected_tv_symbol)
