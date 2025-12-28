import streamlit as st
import streamlit.components.v1 as components

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Monitor B3 | Bollinger & Fundamentalismo",
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

# Tenta importar fundamentus
try:
    import fundamentus
except ImportError:
    st.error("Biblioteca 'fundamentus' não encontrada. Instale com: pip install fundamentus")
    st.stop()

# --- CSS PERSONALIZADO ---
st.markdown("""
    <style>
    /* Cabeçalhos de tabela alinhados */
    [data-testid="stDataFrame"] table tr th { text-align: right !important; }
    [data-testid="stDataFrame"] table tr td { text-align: left !important; }
    
    /* Abas */
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
        # Tenta pegar query params da forma nova ou antiga dependendo da versão do Streamlit
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
            try: st.query_params["visitor_id"] = visitor_id
            except: pass
        
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
        try:
            with open(file_path, "w") as f:
                json.dump(data, f)
        except: pass
            
    return data["total_visits"]

try:
    total_visitantes = update_visitor_counter()
except:
    total_visitantes = 1

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Estatísticas")
    st.metric(label="Visitantes Únicos", value=total_visitantes)
    st.divider()
    st.caption("Scanner B3 Automatizado")

# --- ESTADO DA SESSÃO ---
if 'expander_open' not in st.session_state: st.session_state.expander_open = True
if 'tv_symbol' not in st.session_state: st.session_state.tv_symbol = "BMFBOVESPA:LREN3"

def close_expander(): st.session_state.expander_open = False

# --- FUNÇÕES AUXILIARES DE DADOS ---
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

def apply_best_filters(df):
    if df.empty: return df
    # Filtros Fundamentalistas
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
                # Tenta pegar Net Income
                for name in ['Net Income', 'Net Income Common', 'Lucro Liquido']:
                    if name in fin.index:
                        inc = fin.loc[name]
                        if len(inc) >= 2:
                            curr, prev = inc.iloc[0], inc.iloc[1]
                            if curr < prev:
                                pct = ((curr - prev) / abs(prev)) * 100
                                val_queda = pct
                                lucro_queda_str = f"{pct:.1f}%"
                            else:
                                lucro_queda_str = "Estável/Subiu"
                        break
        except: pass

        if val_queda < 0 or ticker in lista_rj:
            risk_data.append({
                'Ativo': ticker,
                'Preço': row['cotacao'],
                'Alavancagem': row['divbpatr'],
                'Queda Lucro': lucro_queda_str,
                'Situação': status
            })
    return pd.DataFrame(risk_data)

# --- SCANNER BOLLINGER (ABA 2) ---
@st.cache_data(ttl=900)
def scan_bollinger_weekly():
    """
    Retorna apenas ativos cujo candle semanal ATUAL tocou ou furou a banda inferior.
    """
    # Lista de papeis líquidos para escanear
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
        # Baixa dados semanais (interval='1wk')
        data = yf.download(tickers_br, period="2y", interval="1wk", group_by='ticker', progress=False, threads=True)
        
        for t in tickers_br:
            try:
                # Extrai DataFrame do ticker
                if len(tickers_br) > 1:
                    df_t = data[t].copy() if t in data else pd.DataFrame()
                else:
                    df_t = data.copy() # Caso baixe apenas 1
                
                if df_t.empty: continue
                
                # Limpeza básica
                df_t.dropna(subset=['Close'], inplace=True)
                if len(df_t) < 20: continue # Precisa de dados para média de 20

                # Cálculo Bandas de Bollinger (20, 2)
                df_t['SMA20'] = df_t['Close'].rolling(window=20).mean()
                df_t['STD20'] = df_t['Close'].rolling(window=20).std()
                df_t['Lower'] = df_t['SMA20'] - (2.0 * df_t['STD20'])
                df_t['Upper'] = df_t['SMA20'] + (2.0 * df_t['STD20'])
                
                # Pega o candle mais recente (semana atual ou última fechada)
                curr = df_t.iloc[-1]
                
                # --- CRITÉRIO PRINCIPAL ---
                # Se a MÍNIMA (Low) for menor ou igual à Banda Inferior (Lower)
                if curr['Low'] <= curr['Lower']:
                    clean_ticker = t.replace(".SA", "")
                    
                    # Calcula quão longe o fechamento está da banda (negativo = fechou abaixo, positivo = fechou acima)
                    dist = ((curr['Close'] - curr['Lower']) / curr['Lower']) * 100
                    
                    # Define se fechou dentro ou fora
                    situacao = "Furou (Fech. Abaixo)" if curr['Close'] < curr['Lower'] else "Tocou (Fech. Dentro)"
                    
                    candidates.append({
                        'Ativo': clean_ticker,
                        'Preço': curr['Close'],
                        'Mínima': curr['Low'],
                        'Banda Inf': curr['Lower'],
                        'Situação': situacao,
                        'TV_Symbol': f"BMFBOVESPA:{clean_ticker}"
                    })
            except Exception as e:
                continue
        
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- SCANNER ROC EMA (ABA 3) ---
@st.cache_data(ttl=3600*4)
def scan_roc_weekly(df_top_liq):
    if df_top_liq.empty: return pd.DataFrame()
    top_tickers = df_top_liq.sort_values(by='liq2m', ascending=False).head(80)['papel'].tolist()
    tickers_sa = [t + ".SA" for t in top_tickers]
    
    candidates = []
    try:
        data = yf.download(tickers_sa, period="5y", interval="1wk", group_by='ticker', progress=False, threads=True)
        for t in tickers_sa:
            clean_ticker = t.replace(".SA", "")
            try:
                df_t = data[t].copy() if t in data else pd.DataFrame()
                if df_t.empty or len(df_t) < 100: continue
                df_t.dropna(subset=['Close'], inplace=True)
                
                # EMAS
                df_t['EMA17'] = df_t['Close'].ewm(span=17).mean()
                df_t['EMA72'] = df_t['Close'].ewm(span=72).mean()
                
                # ROC simples
                curr = df_t.iloc[-1]
                roc17 = ((curr['Close'] - curr['EMA17']) / curr['EMA17']) * 100
                
                # Lógica: Tendência de alta longa (acima da 72), mas recuo curto (abaixo da 17)
                if (curr['Close'] > curr['EMA72']) and (curr['Close'] < curr['EMA17']):
                    candidates.append({
                        'Ativo': clean_ticker,
                        'Preço': curr['Close'],
                        'ROC17 %': roc17,
                        'TV_Symbol': f"BMFBOVESPA:{clean_ticker}"
                    })
            except: continue
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- WIDGET TRADINGVIEW ---
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
        "theme": "light",
        "style": "1",
        "locale": "br",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": ["MASimple@tv-basicstudies","BB@tv-basicstudies"],
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(html_code, height=500)

# ==========================================
# INTERFACE
# ==========================================
st.title("🇧🇷 Monitor de Mercado B3")
now = datetime.datetime.now()
st.markdown(f"**Data:** {now.strftime('%d/%m/%Y')}")

# Processamento
with st.spinner('Carregando dados...'):
    df_raw = get_ranking_data()
    df_best = apply_best_filters(df_raw)
    df_risk = get_risk_table(df_raw)
    df_bb = scan_bollinger_weekly()
    df_roc = scan_roc_weekly(df_raw)

tab1, tab2, tab3 = st.tabs(["🏆 Fundamentalista", "📉 Bollinger Semanal (Tocou Banda Inf.)", "🚀 Setup ROC (Recuo)"])

# --- ABA 1 ---
with tab1:
    st.subheader("Melhores Ações (Filtro Fundamentalista)")
    if not df_best.empty:
        st.dataframe(df_best[['Ativo','Preço','P/L','ROE','DY','Margem Líq.']].style.format({"Preço":"R${:.2f}", "DY":"{:.1f}%"}), hide_index=True, use_container_width=True)
    else: st.warning("Sem dados fundamentalistas no momento.")
    
    st.divider()
    st.subheader("⚠️ Alerta de Risco (Rec. Judicial ou Dívida Alta + Queda Lucro)")
    if not df_risk.empty:
        st.dataframe(df_risk, hide_index=True, use_container_width=True)

# --- ABA 2: BOLLINGER SEMANAL ---
with tab2:
    st.subheader("📉 Setup: Bandas de Bollinger Semanal (Banda Inferior)")
    st.markdown("Exibe ações onde a **mínima da semana atual** tocou ou rompeu a Banda Inferior de Bollinger (20, 2).")
    
    col_list, col_chart = st.columns([1, 2])
    
    with col_list:
        if not df_bb.empty:
            st.write(f"**{len(df_bb)} Ativos encontrados:**")
            
            # Seletor interativo
            selected_row = st.dataframe(
                df_bb[['Ativo', 'Preço', 'Situação']].style.applymap(
                    lambda v: 'color: red; font-weight: bold' if 'Furou' in v else 'color: green', subset=['Situação']
                ).format({"Preço": "R$ {:.2f}"}),
                use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
            )
            
            # Lógica de seleção para atualizar gráfico
            if len(selected_row.selection.rows) > 0:
                idx = selected_row.selection.rows[0]
                st.session_state.tv_symbol = df_bb.iloc[idx]['TV_Symbol']
            elif st.session_state.tv_symbol == "BMFBOVESPA:LREN3" and not df_bb.empty:
                # Caso padrão se nada selecionado, pega o primeiro da lista
                st.session_state.tv_symbol = df_bb.iloc[0]['TV_Symbol']
        else:
            st.info("Nenhuma ação tocou a banda inferior nesta semana.")
            
    with col_chart:
        clean_name = st.session_state.tv_symbol.split(":")[-1]
        st.markdown(f"#### Gráfico Semanal: {clean_name}")
        show_chart_widget(st.session_state.tv_symbol, interval="W")

# --- ABA 3: ROC ---
with tab3:
    st.subheader("🚀 Setup de Recuo (ROC / Médias)")
    st.markdown("Ações em tendência de alta longa (acima da média 72) fazendo recuo curto (abaixo da média 17).")
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        if not df_roc.empty:
            st.dataframe(df_roc[['Ativo','Preço','ROC17 %']].style.format({"Preço":"R${:.2f}", "ROC17 %":"{:.2f}%"}), hide_index=True, use_container_width=True)
        else: st.info("Nenhum ativo no setup ROC.")
    with col_r2:
        st.info("Selecione um ativo na tabela ao lado (visualização genérica aqui).")
