import streamlit as st
import pandas as pd
import fundamentus
import plotly.graph_objects as go
import feedparser  # BIBLIOTECA PARA LER RSS
import yfinance as yf
import datetime
from datetime import timedelta
from time import mktime

st.markdown("# 🇧🇷 Screener Fundamentalista: As Ações Mais Baratas do Brasil")
st.markdown("### Ranking atualizado de ações com menor P/L, maiores Dividendos e alta eficiência (ROE).")

# --- Configuração da Página ---
st.set_page_config(
    age_title="Ranking de Ações Ibovespa 2025 - Análise Fundamentalista e Dividendos",
    layout="wide",
    page_icon="🇧🇷"
)

# --- CSS Global ---
st.markdown("""
    <style>
    /* Cabeçalhos alinhados à direita */
    [data-testid="stDataFrame"] table tr th {
        text-align: right !important;
    }
    /* Células alinhadas à esquerda */
    [data-testid="stDataFrame"] table tr td {
        text-align: left !important;
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
    """Converte strings financeiras (ex: '10,5%') para float."""
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
        (df['roe'] > 0.05) & 
        (df['pl'] < 15) & (df['pl'] > 0) & 
        (df['evebit'] > 0) & (df['evebit'] < 10) &
        (df['dy'] > 0.04) & 
        (df['mrgliq'] > 0.05) & 
        (df['liq2m'] > 200000)
    )
    df_filtered = df[filtro].copy()
    
    # Ajustes visuais
    df_filtered['dy'] = df_filtered['dy'] * 100
    df_filtered['mrgliq'] = df_filtered['mrgliq'] * 100
    df_filtered['roe'] = df_filtered['roe'] * 100

    df_filtered.rename(columns={
        'papel': 'Ativo', 'cotacao': 'Preço', 'pl': 'P/L', 
        'evebit': 'EV/EBIT', 'dy': 'DY', 'roe': 'ROE', 'mrgliq': 'Margem Líq.'
    }, inplace=True)
    
    # Ordenação padrão
    df_final = df_filtered.sort_values(by=['P/L', 'Margem Líq.'], ascending=[True, False])
    return df_final.reset_index(drop=True)

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
        
        # Anual
        last_3_years = financials.tail(3)
        for date, row in last_3_years.iterrows():
            year_str = str(date.year)
            price = 0.0
            if not hist.empty:
                df_yr = hist[hist.index.year == date.year]
                if not df_yr.empty:
                    price = df_yr['Close'].iloc[-1]
                else:
                    mask = hist.index <= date
                    if mask.any(): price = hist.loc[mask, 'Close'].iloc[-1]
            
            data_rows.append({
                'Periodo': year_str,
                'Receita': row[rev_col],
                'Lucro': row[inc_col],
                'Cotação': price
            })
            
        # TTM
        ttm_rev = 0
        ttm_inc = 0
        has_ttm = False
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
                
            data_rows.append({
                'Periodo': 'Últimos 12m',
                'Receita': ttm_rev,
                'Lucro': ttm_inc,
                'Cotação': curr_price
            })
        
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

# --- Notícias (Horário Brasília UTC-3) ---
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
                    dt_br = dt_utc - timedelta(hours=3) # Ajuste UTC-3
                    date_str = dt_br.strftime("%d/%m %H:%M")
                    dt_obj = dt_br
                except:
                    dt_obj = datetime.datetime.now()
                    date_str = "Recente"

                news_items.append({
                    'title': entry.title,
                    'link': entry.link,
                    'date_obj': dt_obj,
                    'date_str': date_str,
                    'source': source
                })
        except: continue
    news_items.sort(key=lambda x: x['date_obj'], reverse=True)
    return news_items[:6]

# --- Interface Principal ---
st.title("📊 Análise Fundamentalista: Resultados")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")

# 1. Carregamento
with st.spinner('Processando dados...'):
    df_raw = get_ranking_data()
    df_ranking = apply_filters(df_raw)

ith st.expander("ℹ️ Sobre a Metodologia (SEO)", expanded=False):
    st.markdown("""
    **Como encontrar as melhores ações da Bolsa Brasileira (B3)?**
    
    Esta ferramenta realiza uma **análise fundamentalista automática** das ações listadas no Ibovespa e Small Caps. 
    Utilizamos filtros rigorosos para identificar empresas descontadas e rentáveis:
    
    *   **P/L (Preço sobre Lucro):** Buscamos ações baratas com P/L baixo (menor que 15).
    *   **ROE (Retorno sobre o Patrimônio):** Apenas empresas eficientes com ROE acima de 5%.
    *   **Dividend Yield (DY):** Foco em renda passiva com dividendos acima de 4% ao ano.
    *   **Margem Líquida:** Empresas que transformam receita em lucro real.
    
    Os dados são atualizados em tempo real via Yahoo Finance e Fundamentus, focando em ativos como VALE3, PETR4, WEGE3, BBAS3, ITUB4, LREN3, entre outros.
    """)

# 2. Tabela Principal
if not df_ranking.empty:
    st.subheader("🏆 Melhores Ações")
    st.caption("Ordenado por: Menor P/L (Crescente) e Maior Margem Líquida (Decrescente).")
    
    cols_view = ['Ativo', 'Preço', 'EV/EBIT', 'P/L', 'ROE', 'DY', 'Margem Líq.']
    
    # Lógica para Colorir Colunas Pares (2ª, 4ª, 6ª...)
    # Visualmente: Preço(2), P/L(4), DY(6)
    even_cols_subset = ['Preço', 'P/L', 'DY']
    
    # Aplicando estilo: Cinza Claro (#f2f2f2) e Forçando texto Preto (para funcionar no Dark Mode)
    styler = df_ranking[cols_view].style.map(
        lambda x: 'background-color: #f2f2f2; color: black;', 
        subset=even_cols_subset
    ).format({
        "Preço": "R$ {:.2f}",
        "EV/EBIT": "{:.2f}",
        "P/L": "{:.2f}",
        "ROE": "{:.2f}",
        "DY": "{:.2f}",
        "Margem Líq.": "{:.2f}"
    })

    # Configuração adicional do Streamlit
    column_configuration = {
        "Preço": st.column_config.NumberColumn(format="R$ %.2f"),
        "EV/EBIT": st.column_config.NumberColumn(format="%.2f"),
        "P/L": st.column_config.NumberColumn(format="%.2f"),
        "ROE": st.column_config.NumberColumn(format="%.2f"),
        "DY": st.column_config.NumberColumn(format="%.2f"),
        "Margem Líq.": st.column_config.NumberColumn(format="%.2f"),
    }

    st.dataframe(
        styler, 
        use_container_width=True,
        column_config=column_configuration,
        hide_index=True
    )

    st.divider()

    # 3. Gráfico
    st.subheader("📈 Evolução: Cotação vs Lucro vs Receita")
    
    options = df_ranking['Ativo'].tolist()
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

            # EIXO Y1: RECEITA
            fig.add_trace(go.Bar(
                x=df_chart['Periodo'], 
                y=df_chart['Receita'],
                name="Receita", 
                marker=dict(color='#A9A9A9', line=dict(color='black', width=1)),
                text=df_chart['Receita_Texto'],
                textposition='outside',
                yaxis='y1' 
            ))

            # EIXO Y2: LUCRO
            fig.add_trace(go.Scatter(
                x=df_chart['Periodo'], 
                y=df_chart['Lucro'],
                name="Lucro Líquido", 
                mode='lines+markers',
                line=dict(color='#006400', width=3),
                marker=dict(size=8, color='#006400'),
                yaxis='y2' 
            ))

            # EIXO Y3: COTAÇÃO
            fig.add_trace(go.Scatter(
                x=df_chart['Periodo'], 
                y=df_chart['Cotação'],
                name="Cotação", 
                mode='lines+markers',
                line=dict(color='#00008B', width=3),
                marker=dict(size=8, symbol='diamond', color='#00008B'),
                yaxis='y3' 
            ))

            fig.update_layout(
                title=f"{selected}: Análise Visual",
                xaxis=dict(type='category', title="Período"),
                yaxis=dict(title="Receita", side="left", showgrid=False, title_font=dict(color="gray")),
                yaxis2=dict(title="Lucro", side="right", overlaying="y", showgrid=False, title_font=dict(color="green")),
                yaxis3=dict(title="Cotação", side="right", overlaying="y", position=0.95, showgrid=False, showticklabels=False, title_font=dict(color="blue")),
                legend=dict(orientation="h", y=1.1, x=0),
                hovermode="x unified",
                barmode='overlay',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Dados históricos indisponíveis para {selected}.")

else:
    st.warning("Nenhuma ação passou nos filtros atuais.")

st.divider()

# 4. Extras
c1, c2 = st.columns(2)
with c1:
    st.subheader("📰 Notícias (Brasília)")
    news = get_market_news()
    if news:
        for n in news:
            st.markdown(f"**[{n['title']}]({n['link']})**  \n*{n['source']} - {n['date_str']}*")
    else: st.info("Sem notícias.")

with c2:
    st.subheader("💰 Dividendos")
    df_divs = get_latest_dividends(df_ranking['Ativo'].tolist() if not df_ranking.empty else [])
    if not df_divs.empty:
        df_divs['Data'] = df_divs['Data'].dt.strftime('%d/%m/%Y')
        df_divs['Valor'] = df_divs['Valor'].apply(lambda x: f"R$ {x:.4f}")
        st.dataframe(df_divs, hide_index=True)
    else: st.info("Sem dividendos recentes.")

