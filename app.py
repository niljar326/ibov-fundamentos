import streamlit as st
import pandas as pd
import fundamentus
import plotly.graph_objects as go
import feedparser  # BIBLIOTECA PARA LER RSS (NOTÍCIAS REAIS)
import yfinance as yf
import datetime
from time import mktime

# --- Configuração da Página ---
st.set_page_config(
    page_title="Blog Ibovespa - Fundamentalista",
    layout="wide",
    page_icon="🇧🇷"
)

# --- CSS Global: Estilização da Tabela ---
st.markdown("""
    <style>
    /* Cabeçalhos (th) alinhados à DIREITA */
    [data-testid="stDataFrame"] table tr th {
        text-align: right !important;
    }
    
    /* Células de Dados (td) alinhadas à ESQUERDA */
    [data-testid="stDataFrame"] table tr td {
        text-align: left !important;
    }
    
    div[data-testid="stDataFrame"] div[class*="stDataFrame"] {
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# --- Gerenciamento de Estado ---
if 'expander_open' not in st.session_state:
    st.session_state.expander_open = True

def close_expander():
    st.session_state.expander_open = False

# --- Funções Auxiliares ---
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
    if abs_val >= 1e9:
        return f"{val/1e9:.1f}B"
    elif abs_val >= 1e6:
        return f"{val/1e6:.0f}M"
    return f"{val:.0f}"

def get_current_data():
    now = datetime.datetime.now()
    return now.strftime("%B"), now.year

# --- Dados do Ranking ---
@st.cache_data(ttl=3600*6)
def get_ranking_data():
    try:
        df = fundamentus.get_resultado()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'papel'}, inplace=True)
        cols = ['pl', 'roe', 'dy', 'evebit', 'cotacao', 'liq2m', 'mrgliq']
        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_fundamentus_col)
            else:
                df[col] = 0.0
        return df
    except: return pd.DataFrame()

def apply_filters(df):
    if df.empty: return df
    filtro = (
        (df['roe'] > 0.05) & (df['pl'] < 15) & (df['pl'] > 0) & 
        (df['evebit'] > 1) & (df['evebit'] < 9) &
        (df['dy'] > 0.06) & (df['mrgliq'] > 0.05) & (df['liq2m'] > 200000)
    )
    df_filtered = df[filtro].copy()
    
    # Ajustes x100
    df_filtered['dy'] = df_filtered['dy'] * 100
    df_filtered['mrgliq'] = df_filtered['mrgliq'] * 100
    df_filtered['roe'] = df_filtered['roe'] * 100

    df_filtered.rename(columns={
        'papel': 'Ativo', 'cotacao': 'Preço', 'pl': 'P/L', 
        'evebit': 'EV/EBIT', 'dy': 'DY', 'roe': 'ROE', 'mrgliq': 'Margem Líq.'
    }, inplace=True)
    
    return df_filtered.sort_values(by='EV/EBIT', ascending=True)

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

        cols_ref = financials.columns if not financials.empty else []
        rev_col = next((c for c in cols_ref if 'Total Revenue' in c or 'Operating Revenue' in c), None)
        inc_col = next((c for c in cols_ref if 'Net Income' in c or 'Net Income Common' in c), None)
        
        if not rev_col or not inc_col: return None

        data_rows = []
        
        # A: Últimos 3 anos fechados
        last_3_years = financials.tail(3)
        for date, row in last_3_years.iterrows():
            year_str = str(date.year)
            price = 0.0
            if not hist.empty:
                df_yr = hist[hist.index.year == date.year]
                if not df_yr.empty:
                    price = df_yr['Close'].iloc[-1]
            
            data_rows.append({
                'Periodo': year_str,
                'Receita': row[rev_col],
                'Lucro': row[inc_col],
                'Cotação': price
            })
            
        # B: TTM (Últimos 12 Meses)
        if not quarterly.empty and len(quarterly) >= 4:
            last_4_q = quarterly.tail(4)
            ttm_rev = last_4_q[rev_col].sum()
            ttm_inc = last_4_q[inc_col].sum()
            curr_price = 0.0
            if not hist.empty:
                curr_price = hist['Close'].iloc[-1]
                
            data_rows.append({
                'Periodo': 'Últimos 12m',
                'Receita': ttm_rev,
                'Lucro': ttm_inc,
                'Cotação': curr_price
            })
        
        df_final = pd.DataFrame(data_rows)
        df_final['Receita_Texto'] = df_final['Receita'].apply(format_short_number)
        return df_final

    except Exception:
        return None

# --- Dividendos ---
@st.cache_data(ttl=3600*6)
def get_latest_dividends(ticker_list):
    divs_data = []
    # Limita a busca para evitar timeout, pegando os top 10 da lista
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

# --- NOVA FUNÇÃO DE NOTÍCIAS (RSS FEED) ---
@st.cache_data(ttl=1800) # Cache de 30 minutos
def get_market_news():
    # Feeds RSS oficiais das fontes solicitadas
    feeds = {
        'Money Times': 'https://www.moneytimes.com.br/feed/',
        'InfoMoney': 'https://www.infomoney.com.br/feed/',
        'E-Investidor': 'https://einvestidor.estadao.com.br/feed/'
    }
    
    news_items = []
    
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            # Pega as 3 notícias mais recentes de cada fonte
            for entry in feed.entries[:3]:
                # Tenta converter a data de publicação
                try:
                    dt = datetime.datetime.fromtimestamp(mktime(entry.published_parsed))
                    date_str = dt.strftime("%d/%m %H:%M")
                except:
                    date_str = "Recente"

                news_items.append({
                    'title': entry.title,
                    'link': entry.link,
                    'date_obj': dt if 'dt' in locals() else datetime.datetime.now(),
                    'date_str': date_str,
                    'source': source
                })
        except:
            continue
            
    # Ordena todas as notícias por data (mais recente primeiro)
    news_items.sort(key=lambda x: x['date_obj'], reverse=True)
    
    # Retorna as 6 mais recentes no total
    return news_items[:6]

# --- Interface ---
st.title("📊 Análise Fundamentalista: Resultados")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")

# 1. Dados
with st.spinner('Processando dados...'):
    df_raw = get_ranking_data()
    df_ranking = apply_filters(df_raw)

# 2. Tabela Principal
if not df_ranking.empty:
    st.subheader("🏆 Melhores Ações")
    
    cols_view = ['Ativo', 'Preço', 'EV/EBIT', 'P/L', 'ROE', 'DY', 'Margem Líq.']
    
    column_configuration = {
        "Preço": st.column_config.NumberColumn(format="%.2f"),
        "EV/EBIT": st.column_config.NumberColumn(format="%.2f"),
        "P/L": st.column_config.NumberColumn(format="%.2f"),
        "ROE": st.column_config.NumberColumn(format="%.2f"),
        "DY": st.column_config.NumberColumn(format="%.2f"),
        "Margem Líq.": st.column_config.NumberColumn(format="%.2f"),
    }

    st.dataframe(
        df_ranking[cols_view], 
        use_container_width=True,
        column_config=column_configuration,
        hide_index=True
    )

    st.divider()

    # 3. Gráfico
    st.subheader("📈 Evolução: Cotação vs Lucro vs Receita")
    
    options = df_ranking['Ativo'].tolist()
    with st.expander("🔎 Selecionar Ação para o Gráfico", expanded=st.session_state.expander_open):
        selected = st.selectbox("Ativo:", options, on_change=close_expander)

    if selected:
        with st.spinner(f'Gerando gráfico para {selected}...'):
            df_chart = get_chart_data(selected)

        if df_chart is not None and not df_chart.empty:
            
            fig = go.Figure()

            # EIXO Y1: RECEITA
            fig.add_trace(go.Bar(
                x=df_chart['Periodo'], 
                y=df_chart['Receita'],
                name="Receita", 
                marker=dict(
                    color='#696969',
                    line=dict(color='black', width=1.5) 
                ),
                text=df_chart['Receita_Texto'],
                textposition='outside',
                textfont=dict(color='black'),
                yaxis='y1' 
            ))

            # EIXO Y2: LUCRO
            fig.add_trace(go.Scatter(
                x=df_chart['Periodo'], 
                y=df_chart['Lucro'],
                name="Lucro Líquido", 
                mode='lines+markers',
                line=dict(color='#008000', width=3, shape='spline', smoothing=1.3),
                marker=dict(size=9, color='#008000'),
                yaxis='y2' 
            ))

            # EIXO Y3: COTAÇÃO
            fig.add_trace(go.Scatter(
                x=df_chart['Periodo'], 
                y=df_chart['Cotação'],
                name="Cotação", 
                mode='lines+markers',
                line=dict(color='#0000FF', width=3, shape='spline', smoothing=1.3),
                marker=dict(size=9, symbol='diamond', color='#0000FF'),
                yaxis='y3' 
            ))

            fig.update_layout(
                title=f"{selected}: Correlação Visual (Proporcional)",
                xaxis=dict(type='category', title="Período"),
                
                yaxis=dict(
                    title="Receita (R$)", side="left", showgrid=False, title_font=dict(color="#696969")
                ),
                yaxis2=dict(
                    title="Lucro Líquido (R$)", side="right", overlaying="y", showgrid=False,
                    title_font=dict(color="green"), tickfont=dict(color="green")
                ),
                yaxis3=dict(
                    title="Cotação (R$)", side="right", overlaying="y", position=0.95, 
                    showgrid=False, showticklabels=False, title_font=dict(color="blue")
                ),
                legend=dict(orientation="h", y=1.1, x=0),
                hovermode="x unified",
                barmode='overlay',
                margin=dict(t=80)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Nota: 'Últimos 12m' representa o acumulado dos 4 últimos trimestres (TTM).")
        else:
            st.warning("Dados indisponíveis para este ativo.")

else:
    st.warning("Filtros muito restritivos.")

st.divider()

# 4. Notícias e Dividendos
c1, c2 = st.columns(2)

with c1:
    st.subheader("📰 Notícias do Mercado")
    try:
        news = get_market_news()
        if news:
            for n in news:
                # Layout de Notícia
                st.markdown(f"""
                <div style='padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;'>
                    <a href='{n['link']}' target='_blank' style='text-decoration: none; color: #000; font-weight: bold;'>
                        {n['title']}
                    </a>
                    <br>
                    <span style='font-size: 0.8em; color: #555;'>
                        {n['source']} • {n['date_str']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Nenhuma notícia encontrada no momento.")
    except Exception as e:
        st.error(f"Erro ao carregar notícias: {e}")

with c2:
    st.subheader("💰 Dividendos Recentes (Top Filtro)")
    df_divs = get_latest_dividends(df_ranking['Ativo'].tolist() if not df_ranking.empty else [])
    if not df_divs.empty:
        df_divs['Data'] = df_divs['Data'].dt.strftime('%d/%m/%Y')
        df_divs['Valor'] = df_divs['Valor'].apply(lambda x: f"R$ {x:.4f}")
        st.dataframe(df_divs, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum dividendo recente encontrado nos ativos filtrados.")
