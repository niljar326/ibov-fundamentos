import streamlit as st
import streamlit.components.v1 as components # Necessário para o TradingView

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Melhores Ações Ibovespa 2025 | Ranking & Setup Semanal",
    layout="wide",
    page_icon="🇧🇷"
)

# --- IMPORTS GERAIS ---
import pandas as pd
import feedparser # Necessário para as notícias (adicione ao requirements.txt)
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
    /* Ajuste de abas */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

# --- CONTADOR DE VISITANTES ---
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

# --- SCANNER: BANDA DE BOLLINGER SEMANAL (PYTHON) ---
@st.cache_data(ttl=900) # Cache de 15 min para performance
def scan_weekly_bollinger():
    # Lista das ações mais líquidas para verificar (IBOV + Small Caps Líquidas)
    tickers = [
        "VALE3", "PETR4", "ITUB4", "BBDC4", "BBAS3", "PETR3", "ELET3", "WEGE3", "ITSA4", "ABEV3",
        "HAPV3", "BPAC11", "SUZB3", "RDOR3", "EQTL3", "JBSS3", "RADL3", "RAIZ4", "PRIO3", "GGBR4",
        "VBBR3", "VIVT3", "CSAN3", "B3SA3", "BBSE3", "SBSP3", "CMIG4", "MGLU3", "LREN3", "VIIA3",
        "HYPE3", "COGN3", "CPLE6", "CSNA3", "EMBR3", "TIMS3", "EGIE3", "GOAU4", "ENEV3", "ASAI3",
        "CRFB3", "MULT3", "CYRE3", "YDUQ3", "USIM5", "GOLL4", "AZUL4", "TOTS3", "BRFS3", "CVCB3",
        "MRFG3", "FLRY3", "EZTC3", "ALSO3", "JHSF3", "MRVE3", "LWSA3", "SMTO3", "RECV3", "PSSA3"
    ]
    
    tickers_sa = [t + ".SA" for t in tickers]
    
    try:
        # Baixa dados SEMANAIS (interval='1wk')
        data = yf.download(tickers_sa, period="1y", interval="1wk", group_by='ticker', progress=False, threads=True)
        
        candidates = []
        
        for t in tickers:
            try:
                df_t = data[t + ".SA"].copy()
                if df_t.empty: continue
                df_t.dropna(subset=['Close'], inplace=True)
                if len(df_t) < 20: continue

                # Cálculo Matemático das Bandas (Igual ao Pine Script)
                # length = 20, mult = 2.0
                df_t['SMA20'] = df_t['Close'].rolling(window=20).mean()
                df_t['STD20'] = df_t['Close'].rolling(window=20).std()
                df_t['Lower'] = df_t['SMA20'] - (2 * df_t['STD20'])
                
                # Pega a vela ATUAL (Semana corrente)
                current = df_t.iloc[-1]
                
                # LÓGICA: Preço Mínimo tocou ou furou a Banda Inferior?
                # Usamos uma tolerância de 1% (1.01) para pegar toques próximos
                if current['Low'] <= (current['Lower'] * 1.01):
                    dist = ((current['Close'] - current['Lower']) / current['Lower']) * 100
                    
                    candidates.append({
                        'Ativo': t,
                        'Preço Atual': current['Close'],
                        'Banda Inferior': current['Lower'],
                        'Distância %': dist, # Se negativo, fechou abaixo da banda. Se positivo, fechou acima mas tocou.
                        'Data Ref': current.name.strftime('%d/%m')
                    })
            except: continue
            
        df_res = pd.DataFrame(candidates)
        if not df_res.empty:
            return df_res.sort_values('Distância %')
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# --- FUNÇÃO DE NOTÍCIAS (CORRIGIDA) ---
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
                try:
                    dt_utc = datetime.datetime.fromtimestamp(mktime(entry.published_parsed))
                    dt_br = dt_utc - timedelta(hours=3)
                    date_str = dt_br.strftime("%d/%m %H:%M")
                    dt_obj = dt_br
                except: dt_obj, date_str = datetime.datetime.now(), "Recente"
                news_items.append({'title': entry.title, 'link': entry.link, 'date_obj': dt_obj, 'date_str': date_str, 'source': source})
        except: continue
    # Ordena por data
    news_items.sort(key=lambda x: x['date_obj'], reverse=True)
    return news_items[:6]

# --- FUNÇÃO DE DIVIDENDOS (CORRIGIDA) ---
@st.cache_data(ttl=3600*6)
def get_latest_dividends(ticker_list):
    divs_data = []
    # Limita a 10 para não travar o yfinance
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
    
# --- WIDGET TRADINGVIEW DINÂMICO ---
def show_tradingview_widget(ticker):
    # Widget configurado para mostrar BANDAS DE BOLLINGER automaticamente
    symbol = f"BMFBOVESPA:{ticker}"
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 500,
        "symbol": "{symbol}",
        "interval": "W", 
        "timezone": "America/Sao_Paulo",
        "theme": "light",
        "style": "1",
        "locale": "br",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "studies": [
          "BB@tv-basicstudies" 
        ],
        "container_id": "tradingview_widget"
      }});
      </script>
    </div>
    """
    components.html(html_code, height=500)

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================
st.title("🇧🇷 Ranking B3 + Setup BB Semanal")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")

# Carregamento Inicial
with st.spinner('Analisando mercado (Fundamentos + Setup Semanal)...'):
    df_raw = get_ranking_data()
    df_best = apply_best_filters(df_raw)
    df_risk = get_risk_table(df_raw)
    df_bb_weekly = scan_weekly_bollinger() # <--- Executa o Scanner Python

# --- SISTEMA DE ABAS ---
tab1, tab2 = st.tabs(["🏆 Ranking Fundamentalista", "📉 Setup BB Semanal (Oportunidades)"])

# === ABA 1: FUNDAMENTALISTA ===
with tab1:
    st.markdown("### Ações Baratas e Rentáveis (Graham Adaptado)")
    if not df_best.empty:
        styler = df_best[['Ativo', 'Preço', 'P/L', 'ROE', 'DY', 'Margem Líq.']].style.format({
            "Preço": "R$ {:.2f}", "P/L": "{:.2f}", "ROE": "{:.2f}", "DY": "{:.2f}", "Margem Líq.": "{:.2f}"
        })
        st.dataframe(styler, use_container_width=True, hide_index=True)
    else: st.warning("Sem ações no filtro hoje.")

# === ABA 2: SETUP TÉCNICO ===
with tab2:
    st.subheader("📉 Setup: Tocou na Banda Inferior (Semanal)")
    st.markdown("""
    Esta lista mostra ações onde a **Mínima da Semana Atual** tocou ou furou a **Banda de Bollinger Inferior (20, 2)**.
    Isso indica possível exaustão de venda ou oportunidade de repique.
    """)
    
    col_table, col_tv = st.columns([1, 2])
    
    ticker_visualizar = "PETR4" # Default
    
    with col_table:
        if not df_bb_weekly.empty:
            st.write(f"**{len(df_bb_weekly)} ativos encontrados:**")
            
            # Formatação visual
            def highlight_dist(val):
                color = '#ffcccb' if val < 0 else '#e6fffa' # Vermelho claro se furou, Verde claro se só tocou
                return f'background-color: {color}; color: black;'

            st.dataframe(
                df_bb_weekly.style.format({
                    "Preço Atual": "R$ {:.2f}", 
                    "Banda Inferior": "R$ {:.2f}", 
                    "Distância %": "{:.2f}%"
                }).map(highlight_dist, subset=['Distância %']),
                use_container_width=True,
                hide_index=True
            )
            
            # Seletor para mudar o gráfico ao lado
            list_opts = df_bb_weekly['Ativo'].tolist()
            if list_opts:
                ticker_visualizar = st.selectbox("Selecione para ver no Gráfico:", list_opts)
        else:
            st.info("Nenhuma ação tocando a banda inferior nesta semana.")
