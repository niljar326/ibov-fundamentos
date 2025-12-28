import streamlit as st
import streamlit.components.v1 as components # Import necessário para o gráfico TV e Scripts externos

# --- 1. CONFIGURAÇÃO DA PÁGINA (DEVE SER A PRIMEIRA COISA) ---
st.set_page_config(
    page_title="Melhores Ações Ibovespa 2025 | Ranking Fundamentalista e Dividendos",
    layout="wide",
    page_icon="🇧🇷"
)

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

# Tenta importar fundamentus (tratamento de erro caso falhe na nuvem)
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
    
    /* Estilo para o botão de Pix */
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÃO DO CONTADOR DE VISITANTES (ANTI-REFRESH) ---
def update_visitor_counter():
    file_path = "visitor_counter.json"
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 1. Gerenciamento de ID via URL (Query Params)
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
        if hasattr(st, "query_params"):
            st.query_params["visitor_id"] = visitor_id
        
    # 2. Gerenciamento do Arquivo JSON
    data = {"total_visits": 0, "daily_visits": {}}

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except:
            pass 

    if today not in data["daily_visits"]:
        data["daily_visits"][today] = []

    # 3. Contagem
    if visitor_id not in data["daily_visits"][today]:
        data["daily_visits"][today].append(visitor_id)
        data["total_visits"] += 1
        
        with open(file_path, "w") as f:
            json.dump(data, f)
            
    return data["total_visits"]

# --- Executa o Contador ---
try:
    total_visitantes = update_visitor_counter()
except Exception as e:
    total_visitantes = 0 

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("📊 Estatísticas")
    st.metric(label="Visitantes Únicos", value=total_visitantes, help="Visitantes únicos (não conta F5)")
    st.divider()
            
    st.caption("Desenvolvido com Streamlit")

# --- Estado ---
if 'expander_open' not in st.session_state: st.session_state.expander_open = True
if 'tv_symbol' not in st.session_state: st.session_state.tv_symbol = "BMFBOVESPA:LREN3" # Padrão inicial

def close_expander(): st.session_state.expander_open = False

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
            if col in df.columns:
                df[col] = df[col].apply(clean_fundamentus_col)
            else:
                df[col] = 0.0
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

# --- LÓGICA DA TABELA DE RISCO ---
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
        status = "Recup. Judicial / Reestruturação" if ticker in lista_rj else "Alta Alavancagem"
        lucro_queda_str = "N/D"
        val_queda = 0.0
        
        try:
            stock = yf.Ticker(ticker + ".SA")
            fin = stock.financials
            if not fin.empty:
                inc_row = None
                possible_names = ['Net Income', 'Net Income Common', 'Net Income Continuous']
                for name in possible_names:
                    if name in fin.index:
                        inc_row = fin.loc[name]
                        break
                
                if inc_row is not None and len(inc_row) >= 2:
                    curr_profit = inc_row.iloc[0]
                    prev_profit = inc_row.iloc[1]
                    if curr_profit < prev_profit:
                        diff = (curr_profit - prev_profit)
                        pct = (diff / abs(prev_profit)) * 100
                        val_queda = pct 
                        lucro_queda_str = f"{pct:.1f}%"
                    else:
                        lucro_queda_str = "Subiu/Estável"
                else:
                    lucro_queda_str = "Sem Hist."
        except:
            lucro_queda_str = "Erro dados"

        if val_queda < 0 or ticker in lista_rj:
            risk_data.append({
                'Ativo': ticker,
                'Preço': row['cotacao'],
                'Alavancagem (Dív/Patr)': row['divbpatr'],
                'Queda Lucro (Ano)': lucro_queda_str,
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
            if not d.empty:
                divs_data.append({'Ativo': ticker, 'Valor': d.iloc[-1], 'Data': d.index[-1]})
        except: continue
    if divs_data:
        df = pd.DataFrame(divs_data)
        df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
        return df.sort_values('Data', ascending=False).head(5)
    return pd.DataFrame()

# --- Notícias ---
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
    news_items.sort(key=lambda x: x['date_obj'], reverse=True)
    return news_items[:6]

# --- SCANNER BOLLINGER (SÓ BRASIL - SEMANAL - SÓ LOWER BAND) ---
@st.cache_data(ttl=600)
def scan_bollinger_weekly():
    # Lista de Ações Líquidas da B3
    tickers_br = [
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "WEGE3.SA", "PRIO3.SA", 
        "MGLU3.SA", "HAPV3.SA", "RDOR3.SA", "SUZB3.SA", "JBSS3.SA", "RAIZ4.SA", "GGBR4.SA", "CSAN3.SA",
        "VBBR3.SA", "B3SA3.SA", "BBSE3.SA", "CMIG4.SA", "ITSA4.SA", "BHIA3.SA", "GOLL4.SA", "AZUL4.SA", 
        "CVCB3.SA", "USIM5.SA", "CSNA3.SA", "EMBR3.SA", "CPLE6.SA", "RADL3.SA", "EQTL3.SA", "TOTS3.SA", 
        "RENT3.SA", "TIMS3.SA", "SBSP3.SA", "ELET3.SA", "ABEV3.SA", "ASAI3.SA", "CRFB3.SA", "MULT3.SA",
        "CYRE3.SA", "EZTC3.SA", "MRVE3.SA", "PETZ3.SA", "SOMA3.SA", "ALPA4.SA", "LREN3.SA"
    ]
    
    candidates = []
    
    try:
        # Baixa dados SEMANAIS ('1wk') dos últimos 2 anos (necessário para calcular SMA 20 semanas)
        data = yf.download(tickers_br, period="2y", interval="1wk", group_by='ticker', progress=False, threads=True)
        
        for t in tickers_br:
            try:
                df_t = data[t].copy() if t in data else pd.DataFrame()
                if df_t.empty: continue
                
                # Limpeza
                df_t.dropna(subset=['Close'], inplace=True)
                if len(df_t) < 22: continue # Precisa de histórico para a média

                # Cálculo Bandas Bollinger (20, 2)
                # Basis = SMA 20
                df_t['SMA20'] = df_t['Close'].rolling(window=20).mean()
                # StdDev
                df_t['STD20'] = df_t['Close'].rolling(window=20).std()
                # Lower Band = Basis - 2 * StdDev
                df_t['Lower'] = df_t['SMA20'] - (2.0 * df_t['STD20'])
                
                # Pega a vela ATUAL (Semana corrente)
                curr = df_t.iloc[-1]
                
                # CRITÉRIO: Mínima da semana tocou ou furou a Banda Inferior
                if curr['Low'] <= curr['Lower']:
                    clean_ticker = t.replace(".SA", "")
                    
                    dist = ((curr['Close'] - curr['Lower']) / curr['Lower']) * 100
                    
                    candidates.append({
                        'Ativo': clean_ticker,
                        'Preço Atual': curr['Close'],
                        'Mínima Sem.': curr['Low'],
                        'Banda Inf': curr['Lower'],
                        'Distância Fech %': dist,
                        'TV_Symbol': f"BMFBOVESPA:{clean_ticker}"
                    })
            except: continue
        
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- SCANNER ROC EMA (SETUP CAIU COMPROU) ---
@st.cache_data(ttl=3600*4)
def scan_roc_weekly(df_top_liq):
    # Pega as top 90 ações por liquidez do dataframe fundamentalista
    if df_top_liq.empty: return pd.DataFrame()
    
    # Ordena por liquidez e pega top 90 (simulando top ibov)
    top_tickers = df_top_liq.sort_values(by='liq2m', ascending=False).head(90)['papel'].tolist()
    
    # Adiciona sufixo .SA
    tickers_sa = [t + ".SA" for t in top_tickers]
    
    candidates = []
    
    try:
        # Baixa dados SEMANAIS ('1wk') dos últimos 6 anos (precisa de 305 periodos para ema4)
        data = yf.download(tickers_sa, period="7y", interval="1wk", group_by='ticker', progress=False, threads=True)
        
        for t in tickers_sa:
            clean_ticker = t.replace(".SA", "")
            try:
                df_t = data[t].copy() if t in data else pd.DataFrame()
                if df_t.empty: continue
                
                df_t.dropna(subset=['Close'], inplace=True)
                # Verifica histórico suficiente para EMA 305 (305 semanas)
                if len(df_t) < 310: continue

                # Cálculo das EMAS
                # EMA 17
                df_t['EMA17'] = df_t['Close'].ewm(span=17, adjust=False).mean()
                # EMA 34
                df_t['EMA34'] = df_t['Close'].ewm(span=34, adjust=False).mean()
                # EMA 72
                df_t['EMA72'] = df_t['Close'].ewm(span=72, adjust=False).mean()
                # EMA 305
                df_t['EMA305'] = df_t['Close'].ewm(span=305, adjust=False).mean()
                
                # ROC (Distância % do Preço para a EMA)
                # ROC = ((Close - EMA) / EMA) * 100
                df_t['ROC17'] = ((df_t['Close'] - df_t['EMA17']) / df_t['EMA17']) * 100
                df_t['ROC34'] = ((df_t['Close'] - df_t['EMA34']) / df_t['EMA34']) * 100
                df_t['ROC72'] = ((df_t['Close'] - df_t['EMA72']) / df_t['EMA72']) * 100
                df_t['ROC305'] = ((df_t['Close'] - df_t['EMA305']) / df_t['EMA305']) * 100
                
                curr = df_t.iloc[-1]
                
                # LÓGICA DO USUÁRIO
                # 1. EMA1 Negativa, resto Positiva -> "Alta (Caiu Comprou)"
                cond_alta = (
                    (curr['ROC17'] < 0) & 
                    (curr['ROC34'] > 0) & 
                    (curr['ROC72'] > 0) & 
                    (curr['ROC305'] > 0)
                )
                
                # 2. EMA1 Negativa, EMA2 Maior que EMA1 (mesmo que negativa), resto Positiva -> "Média"
                # A condição "ema2 maior que ema1" matematicamente é roc34 > roc17.
                # Se roc34 for positivo, cai na regra acima (se o resto for positivo).
                # Se roc34 for negativo, cai nesta regra se for maior que roc17.
                cond_media = (
                    (curr['ROC17'] < 0) &
                    (curr['ROC34'] < 0) &
                    (curr['ROC34'] > curr['ROC17']) &
                    (curr['ROC72'] > 0) &
                    (curr['ROC305'] > 0)
                )
                
                probabilidade = ""
                if cond_alta:
                    probabilidade = "Alta (Caiu Comprou)"
                elif cond_media:
                    probabilidade = "Média"
                
                if probabilidade:
                    candidates.append({
                        'Ativo': clean_ticker,
                        'Preço': curr['Close'],
                        'Probabilidade': probabilidade,
                        'ROC17 %': curr['ROC17'],
                        'ROC34 %': curr['ROC34'],
                        'ROC305 %': curr['ROC305'],
                        'TV_Symbol': f"BMFBOVESPA:{clean_ticker}"
                    })

            except: continue
            
        return pd.DataFrame(candidates)
            
    except: return pd.DataFrame()

# --- WIDGET CHART TRADINGVIEW ---
def show_chart_widget(symbol_tv):
    # Widget configurado para Semanal ("W") com o Estudo "BB@tv-basicstudies" (Bollinger Bands)
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
        "interval": "W", 
        "timezone": "America/Sao_Paulo",
        "theme": "light",
        "style": "1",
        "locale": "br",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "MASimple@tv-basicstudies", 
            "MASimple@tv-basicstudies",
            "MASimple@tv-basicstudies"
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

# --- BANNER TOPO (CENTRALIZADO E PEQUENO) ---
components.html("""
    <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
        <script src="https://pl28325401.effectivegatecpm.com/1a/83/79/1a8379a4a8ddb94a327a5797257a9f02.js"></script>
    </div>
""", height=90)
# --------------------------------------------

st.title("🇧🇷 Ranking de Ações Baratas e Rentáveis - B3")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")

# --- AVISO LEGAL (NOVO) ---
st.warning("⚠️ **AVISO IMPORTANTE:** As informações disponibilizadas nesta página não configuram recomendação de compra ou venda. O mercado financeiro possui riscos. Utilize os dados apenas para estudo e aprofunde sua análise antes de tomar qualquer decisão.")

# 1. Carregamento dos Dados
with st.spinner('Processando dados do mercado...'):
    df_raw = get_ranking_data()
    df_best = apply_best_filters(df_raw)
    df_warning = get_risk_table(df_raw)
    df_scan_bb = scan_bollinger_weekly() # Novo Scanner BB
    df_scan_roc = scan_roc_weekly(df_raw) # Novo Scanner ROC

# --- SISTEMA DE ABAS ---
tab1, tab2, tab3 = st.tabs(["🏆 Ranking Fundamentalista", "📉 Setup BB Semanal (Lower Band)", "🚀 Setup ROC (Caiu Comprou)"])

# === ABA 1: CONTEÚDO ORIGINAL ===
with tab1:
    st.markdown("""
    <div style="text-align: justify; margin-bottom: 20px;">
    Este <b>Screener Fundamentalista</b> filtra automaticamente as melhores oportunidades. 
    Abaixo, você também encontra uma lista de <b>Alerta</b> para empresas em situações delicadas.
    </div>
    """, unsafe_allow_html=True)

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

        st.dataframe(styler, use_container_width=True, column_config={"Preço": st.column_config.NumberColumn(format="R$ %.2f")}, hide_index=True)

    st.divider()
    col_ad1, col_ad2 = st.columns(2)
    with col_ad1:
        st.markdown("""
        <div style="background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 10px; color: #333; height: 100%;">
            <h4 style="margin-top:0; color: #333;">✈️ Nomad: Taxa Zero em Dólar</h4>
            <p style="font-size: 14px;">Ganhe taxa zero na 1ª conversão (até US$ 1.000) para investir nos EUA.</p>
            <p style="font-size: 14px;">Código: <code style="background-color: #eee; padding: 4px; border-radius: 4px; border: 1px solid #ddd; font-weight:bold;">Y39FP3XF8I</code></p>
            <div style="text-align:center;"><a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I%26n=Jader" target="_blank" style="text-decoration: none; color: white; background-color: #1a1a1a; padding: 10px 15px; border-radius: 5px; font-size: 14px; display: inline-block; width: 100%;">➡️ <b>Abrir Conta Nomad</b></a></div>
        </div>
        """, unsafe_allow_html=True)
    with col_ad2:
        st.markdown("""
        <div style="background-color: #eaf6ff; border: 1px solid #bae0ff; padding: 15px; border-radius: 10px; color: #333; height: 100%;">
            <h4 style="margin-top:0; color: #009ee3;">🤝 Mercado Pago: R$ 30 OFF</h4>
            <p style="font-size: 14px;">Use o app pela primeira vez (pagamento mín. R$ 70) e ganhe <b>R$ 30 de desconto</b>.</p>
            <div style="text-align:center;"><a href="https://mpago.li/1VydVhw" target="_blank" style="text-decoration: none; color: white; background-color: #009ee3; padding: 10px 15px; border-radius: 5px; font-size: 14px; display: inline-block; width: 100%;">➡️ <b>Resgatar R$ 30</b></a></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("⚠️ Atenção! Empresas em Risco / Recup. Judicial")
    st.markdown("**Critérios:** Em Recuperação Judicial (Lista B3) **OU** Alavancagem Alta (Dívida > 3x Patrimônio) **E** Queda no Lucro.")
    if not df_warning.empty:
        def color_negative_red(val):
            if isinstance(val, str) and '-' in val: return 'color: red; font-weight: bold;'
            return ''
        styler_risk = df_warning.style.map(color_negative_red, subset=['Queda Lucro (Ano)']).format({"Preço": "R$ {:.2f}", "Alavancagem (Dív/Patr)": "{:.2f}"})
        st.dataframe(styler_risk, use_container_width=True, hide_index=True)
    else: st.info("Nenhuma ação com os critérios de risco (Dívida Extrema + Queda Lucro) encontrada hoje.")

    st.divider()
    st.subheader("📈 Análise Visual: Cotação vs Lucro")
    options = df_best['Ativo'].tolist()
    idx_default = 0
    if 'LREN3' in options:
        try: idx_default = options.index('LREN3')
        except: pass
    with st.expander("🔎 Selecionar Ação para o Gráfico", expanded=st.session_state.expander_open):
        selected = st.selectbox("Ativo:", options, index=idx_default, on_change=close_expander)
    if selected:
        with st.spinner(f'Gerando gráfico para {selected}...'):
            df_chart = get_chart_data(selected)
        if df_chart is not None and not df_chart.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_chart['Periodo'], y=df_chart['Receita'], name="Receita", marker=dict(color='#A9A9A9', line=dict(color='black', width=1)), text=df_chart['Receita_Texto'], textposition='outside', yaxis='y1'))
            fig.add_trace(go.Scatter(x=df_chart['Periodo'], y=df_chart['Lucro'], name="Lucro Líquido", mode='lines+markers', line=dict(color='#006400', width=3), marker=dict(size=8, color='#006400'), yaxis='y2'))
            fig.add_trace(go.Scatter(x=df_chart['Periodo'], y=df_chart['Cotação'], name="Cotação", mode='lines+markers', line=dict(color='#00008B', width=3), marker=dict(size=8, symbol='diamond', color='#00008B'), yaxis='y3'))
            fig.update_layout(title=f"{selected}: Receita vs Lucro vs Preço", xaxis=dict(type='category', title="Período"), yaxis=dict(title="Receita", side="left", showgrid=False, title_font=dict(color="gray")), yaxis2=dict(title="Lucro", side="right", overlaying="y", showgrid=False, title_font=dict(color="green")), yaxis3=dict(title="Cotação", side="right", overlaying="y", position=0.95, showgrid=False, showticklabels=False, title_font=dict(color="blue")), legend=dict(orientation="h", y=1.1, x=0), hovermode="x unified", barmode='overlay', height=500)
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning(f"Dados históricos indisponíveis para {selected}.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📰 Notícias (Brasília)")
        news = get_market_news()
        if news:
            for n in news: st.markdown(f"**[{n['title']}]({n['link']})**  \n*{n['source']} - {n['date_str']}*")
        else: st.info("Sem notícias.")
    with c2:
        st.subheader("💰 Dividendos Recentes")
        df_divs = get_latest_dividends(df_best['Ativo'].tolist() if not df_best.empty else [])
        if not df_divs.empty:
            df_divs['Data'] = df_divs['Data'].dt.strftime('%d/%m/%Y')
            df_divs['Valor'] = df_divs['Valor'].apply(lambda x: f"R$ {x:.4f}")
            st.dataframe(df_divs, hide_index=True)
        else:
            st.info("Sem dividendos recentes.")

# === ABA 2: SCANNER BB (SÓ BRASIL - SEMANAL - SÓ LOWER) ===
with tab2:
    st.subheader("📉 Setup: Bandas de Bollinger Semanal (Lower Band)")
    st.warning("⚠️ **Atenção:** Este filtro mostra ações tocando a banda inferior. Considere o fato de que ações em forte tendência de baixa podem continuar caindo.")
    
    col_list, col_chart = st.columns([1, 2])
    
    with col_list:
        if not df_scan_bb.empty:
            st.write(f"**{len(df_scan_bb)} Oportunidades:**")
            event = st.dataframe(
                df_scan_bb[['Ativo', 'Preço Atual', 'Distância Fech %']].style.format({
                    "Preço Atual": "{:.2f}", "Distância Fech %": "{:.2f}%"
                }).map(lambda x: 'background-color: #ffcccb; color: black', subset=['Distância Fech %']),
                use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
            )
            if len(event.selection.rows) > 0:
                selected_index = event.selection.rows[0]
                st.session_state.tv_symbol = df_scan_bb.iloc[selected_index]['TV_Symbol']
        else:
            st.info("Nenhuma ação brasileira encontrada tocando a banda inferior nesta semana.")
            
    with col_chart:
        clean_name = st.session_state.tv_symbol.split(":")[-1]
        st.markdown(f"#### Gráfico Semanal: {clean_name}")
        show_chart_widget(st.session_state.tv_symbol)

# === ABA 3: NOVO SCANNER ROC (EMA 17/34/72/305) ===
with tab3:
    st.subheader("🚀 Setup ROC: Médias Exponenciais (Semanal)")
    st.markdown("""
    **Conceito (Caiu Comprou):** Busca ações em tendência primária de alta (acima das EMAs 72 e 305) que fizeram um recuo (pullback) abaixo das médias curtas.
    *   **Alta Probabilidade:** Preço abaixo da EMA17, mas acima das demais.
    *   **Média Probabilidade:** Preço abaixo da EMA17 e EMA34, mas a EMA34 ainda está acima da EMA17 (ordem preservada) e acima das longas.
    """)
    
    col_roc_list, col_roc_chart = st.columns([1, 2])

    with col_roc_list:
        if not df_scan_roc.empty:
            st.write(f"**{len(df_scan_roc)} Ativos Encontrados (Top Liquidez):**")
            
            # Formatação condicional da coluna Probabilidade
            def color_prob(val):
                color = '#d4edda' if 'Alta' in val else '#fff3cd'
                return f'background-color: {color}; color: black; font-weight: bold;'

            event_roc = st.dataframe(
                df_scan_roc[['Ativo', 'Preço', 'Probabilidade', 'ROC17 %']].style.format({
                    "Preço": "R$ {:.2f}", "ROC17 %": "{:.2f}%"
                }).map(color_prob, subset=['Probabilidade']),
                use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
            )
            
            if len(event_roc.selection.rows) > 0:
                idx_roc = event_roc.selection.rows[0]
                st.session_state.tv_symbol = df_scan_roc.iloc[idx_roc]['TV_Symbol']
            elif st.session_state.tv_symbol == "BMFBOVESPA:LREN3" and not df_scan_roc.empty:
                 pass
        else:
            st.info("Nenhuma ação do Top Liquidez atende aos critérios ROC nesta semana.")

    with col_roc_chart:
        clean_name_roc = st.session_state.tv_symbol.split(":")[-1]
        st.markdown(f"#### Gráfico Semanal: {clean_name_roc}")
        # Reutiliza o widget chart (o estudo aplicado será padrão, pois o widget não aceita input dinâmico de script pine complexo via URL simples, mas mostrará o ativo)
        show_chart_widget(st.session_state.tv_symbol)
