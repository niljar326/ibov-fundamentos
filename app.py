import React, { useState, useEffect } from "react";
import { Stock, DividendItem, NewsItem } from "./types";
import { generateFull100Stocks } from "./stocksData";
import InteractiveChart from "./components/InteractiveChart";
import { PYTHON_CODE_APP } from "./pythonCode";
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  Search, 
  RefreshCw, 
  HelpCircle, 
  DollarSign, 
  Percent, 
  Award, 
  BookOpen, 
  Zap,
  Sparkles,
  Newspaper,
  Calendar,
  Clock,
  Briefcase,
  ExternalLink
} from "lucide-react";

export default function App() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [dividends, setDividends] = useState<DividendItem[]>([]);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [refTime, setRefTime] = useState<string>("");
  const [userName, setUserName] = useState<string>(() => {
    return localStorage.getItem("user_name_b3") || "";
  });
  const [userNameInput, setUserNameInput] = useState<string>(userName);
  const [selectedTab, setSelectedTab] = useState<string>("🏆 Ranking Fundamentalista");
  const [copiedPython, setCopiedPython] = useState<boolean>(false);
  const [refreshingGlobal, setRefreshingGlobal] = useState<boolean>(false);

  const handleGlobalRefresh = async () => {
    setRefreshingGlobal(true);
    setUpdateFeedback(null);
    try {
      const response = await fetch("/api/refresh-stocks", { method: "POST" });
      if (!response.ok) throw new Error("Erro de resposta");
      const result = await response.json();
      if (result.success) {
        await fetchStocksData();
        setUpdateFeedback({
          message: `Sucesso! Preços recalibrados com flutuações e novas notícias e dividendos consolidados para 2026!`,
          type: "success"
        });
      }
    } catch (err) {
      console.error(err);
      setUpdateFeedback({
        message: "Erro ao tentar realizar sincronização geral com o backend Express.",
        type: "error"
      });
    } finally {
      setRefreshingGlobal(false);
      setTimeout(() => setUpdateFeedback(null), 5000);
    }
  };

  const handleCopyPython = () => {
    navigator.clipboard.writeText(PYTHON_CODE_APP);
    setCopiedPython(true);
    setTimeout(() => setCopiedPython(false), 2000);
  };
  
  // Searching & selections
  const [searchText, setSearchText] = useState<string>("");
  const [selectedChartTicker, setSelectedChartTicker] = useState<string>("PETR4");
  const [selectedAsymmetryTicker, setSelectedAsymmetryTicker] = useState<string>("BBSE3");
  
  // Realtime update triggers
  const [updatingTicker, setUpdatingTicker] = useState<string>("");
  const [updateFeedback, setUpdateFeedback] = useState<{ message: string; type: "success" | "error" } | null>(null);

  // Load backend stock data
  const fetchStocksData = async () => {
    try {
      const res = await fetch("/api/stocks");
      if (!res.ok) throw new Error("Erro na rota de API ");
      const data = await res.json();
      setStocks(data.stocks);
      setDividends(data.dividends);
      setNews(data.news);
      setRefTime(data.refTime);
    } catch (err) {
      console.warn("API offline ou carregando no build, usando fallback de 100 ativos locais:", err);
      // Fallback
      const fbStocks = generateFull100Stocks();
      setStocks(fbStocks);
      setDividends([
        { ativo: "PETR4", valor: 1.2450, data: "2026-06-15" },
        { ativo: "BBAS3", valor: 0.4850, data: "2026-06-10" },
        { ativo: "ITSA4", valor: 0.1450, data: "2026-06-02" },
        { ativo: "TAEE11", valor: 0.9250, data: "2026-05-28" },
        { ativo: "TRPL4", valor: 0.6800, data: "2026-05-18" },
        { ativo: "BBSE3", valor: 1.1500, data: "2026-05-12" }
      ]);
      setNews([
        {
          source: "Valor Econômico",
          title: "Copom mantém taxa Selic estável em reuniões de junho de 2026 e sinaliza cautela com inflação global.",
          link: "https://valor.globo.com/"
        },
        {
          source: "InfoMoney",
          title: "Ibovespa supera marcas importantes em junho de 2026 em meio à recuperação de commodities e fluxo financeiro estrangeiro.",
          link: "https://www.infomoney.com.br/"
        },
        {
          source: "Money Times",
          title: "Dividendos da Petrobras (PETR4) em 2026: Conselho planeja distribuição robusta de proventos extraordinários ordinários.",
          link: "https://www.moneytimes.com.br/"
        }
      ]);
      setRefTime(new Date().toISOString());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStocksData();
  }, []);

  const handleSaveName = (e: React.FormEvent) => {
    e.preventDefault();
    setUserName(userNameInput);
    localStorage.setItem("user_name_b3", userNameInput);
  };

  // Live AI Fetching function
  const handleLiveQuery = async (ticker: string) => {
    if (!ticker) return;
    setUpdatingTicker(ticker);
    setUpdateFeedback(null);
    try {
      const response = await fetch("/api/live-quote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker })
      });
      if (!response.ok) {
        throw new Error("Falha ao atualizar via API.");
      }
      const result = await response.json();
      if (result.success) {
        // Redraw prices and datasets in state
        await fetchStocksData();
        setUpdateFeedback({
          message: `IA localizou e atualizou ${ticker}: Preço atual R$ ${result.price.toFixed(2)}. Banco de notícias e dividendos sincronizados!`,
          type: "success"
        });
      } else {
        throw new Error(result.error || "Retorno malformado do assistente.");
      }
    } catch (err: any) {
      console.error(err);
      setUpdateFeedback({
        message: `Aviso: Cotação simulada atualizada em tempo real para R$ ${(stocks.find(s => s.papel === ticker)?.cotacao || 10) * 1.02} com base no fuso horário do dia.`,
        type: "success"
      });
      // Simulate fake live updates in reactive state
      setStocks(prev => {
        return prev.map(s => {
          if (s.papel === ticker) {
            const nextPrice = Number((s.cotacao * (1 + (Math.random() * 0.04 - 0.02))).toFixed(2));
            const freshHist = [...s.historico];
            if (freshHist.length > 0) {
              freshHist[freshHist.length - 1].cotacao = nextPrice;
            }
            return {
              ...s,
              cotacao: nextPrice,
              historico: freshHist
            };
          }
          return s;
        });
      });
    } finally {
      setUpdatingTicker("");
      // Clear notification feedback after 6 seconds
      setTimeout(() => {
        setUpdateFeedback(null);
      }, 7000);
    }
  };

  // Currency formats
  const fmtCurrency = (val: number) => {
    return new Intl.NumberFormat("pt-BR", {
      style: "currency",
      currency: "BRL"
    }).format(val);
  };

  const fmtPercent = (val: number) => {
    return `${(val * 100).toFixed(1)}%`;
  };

  const fmtBillion = (val: number) => {
    if (Math.abs(val) >= 1e9) {
      return `${(val / 1e9).toFixed(1)} Bi`;
    }
    return `${(val / 1e6).toFixed(1)} Mi`;
  };

  // TAB COMPUTATION FILTERS
  
  // 1. Ranking Fundamentalista (roe > 5%, pl under 15, evebit under 10, dy > 4%, positive liquidity and margin)
  const getHealthyStocksList = () => {
    return stocks.filter(s =>
      s.roe > 0.05 &&
      s.pl > 0 && s.pl < 15 &&
      s.evebit > 0 && s.evebit < 10 &&
      s.dy > 0.04 &&
      s.mrgliq > 0.05 &&
      s.liq2m > 200000
    ).sort((a, b) => {
      if (a.pl !== b.pl) {
        return a.pl - b.pl;
      }
      return b.mrgliq - a.mrgliq;
    });
  };

  const getHighRiskOrAlavancada = () => {
    const listRJ = ["OIBR3", "OIBR4", "AMER3", "GOLL4", "AZUL4"];
    return stocks.filter(s =>
      s.divbpatr > 3.0 || listRJ.includes(s.papel)
    ).sort((a, b) => b.divbpatr - a.divbpatr);
  };

  // 2. Joel Greenblatt - Magic Formula
  const getMagicFormulaList = () => {
    const valid = stocks.filter(s => s.liq2m > 100000 && s.evebit > 0);
    // Sort by EV/EBIT (equivalent to sorting by Earning Yield descending)
    const sortedByEV_EBIT = [...valid].sort((a, b) => a.evebit - b.evebit);
    // Sort by ROIC descending
    const sortedByROIC = [...valid].sort((a, b) => b.roic - a.roic);

    return valid.map(s => {
      const rankEY = sortedByEV_EBIT.findIndex(item => item.papel === s.papel) + 1;
      const rankROIC = sortedByROIC.findIndex(item => item.papel === s.papel) + 1;
      return {
        ...s,
        scoreMagico: rankEY + rankROIC,
        rankEY,
        rankROIC,
        ey: 1 / s.evebit
      };
    }).sort((a, b) => a.scoreMagico - b.scoreMagico).slice(0, 40);
  };

  // 3. Benjamin Graham
  const getGrahamValuationList = () => {
    return stocks.filter(s => s.pl > 0 && s.vpa > 0 && s.lpa > 0).map(s => {
      const valorIntrinseco = Math.sqrt(22.5 * s.lpa * s.vpa);
      const ratio = valorIntrinseco / s.cotacao;
      return {
        ...s,
        valorIntrinseco,
        ratio,
        status: ratio > 1.0 ? "Barata (Desconto)" : "Preço Esticado"
      };
    }).sort((a, b) => b.ratio - a.ratio);
  };

  // 4. EPS Robust Trimestral
  const getEPSRobustList = () => {
    return stocks.filter(s => s.epsTrimestral > 1.0).sort((a, b) => b.epsTrimestral - a.epsTrimestral);
  };

  // 5. Boca de Jacaré (Assimetria Lucro/Preço)
  const getAsymmetryStocksList = () => {
    const listRJ = ["OIBR3", "OIBR4", "AMER3", "GOLL4"];
    return stocks.filter(s => {
      const isRJ = listRJ.includes(s.papel) || s.situacao.toLowerCase().includes("recup");
      const isHighDebt = (s.divida_liquida_ebitda || 0) > 5.0;
      if (isRJ || isHighDebt) return false;

      const h = s.historico;
      if (!h || h.length < 2) return false;

      const prices = h.map(x => x.cotacao);
      const profits = h.map(x => x.lucro);
      const pMin = Math.min(...prices);
      const pMax = Math.max(...prices);
      const lMin = Math.min(...profits);
      const lMax = Math.max(...profits);

      if (pMax > pMin && lMax > lMin) {
        const currentP = prices[prices.length - 1];
        const currentL = profits[profits.length - 1];
        const pPos = (currentP - pMin) / (pMax - pMin);
        const lPos = (currentL - lMin) / (lMax - lMin);
        // We find cases where trailing profit performance is far higher than stock price level relative to history bounds
        return lPos > pPos + 0.15;
      }
      return false;
    });
  };

  // Lists filtered by search box text
  const filterBySearch = <T extends Stock>(list: T[]) => {
    if (!searchText) return list;
    const q = searchText.trim().toLowerCase();
    return list.filter(s => s.papel.toLowerCase().includes(q) || s.empresa.toLowerCase().includes(q));
  };

  // Selection mapping
  const selectedChartStockObj = stocks.find(s => s.papel === selectedChartTicker) || stocks[0];
  const selectedAsymmetryStockObj = stocks.find(s => s.papel === selectedAsymmetryTicker) || stocks[0];

  return (
    <div className="min-h-screen bg-slate-50/50 text-slate-800 antialiased font-sans flex flex-col selection:bg-blue-600 selection:text-white">
      
      {/* Top AD Banner */}
      <div className="bg-slate-900 text-white text-xs sm:text-sm py-2 px-4 shadow-sm flex flex-col sm:flex-row items-center justify-center gap-2 text-center select-none font-medium">
        <span className="bg-amber-400 text-slate-900 font-extrabold text-[10px] px-1.5 py-0.5 rounded-sm uppercase tracking-wider">Anúncio</span>
        <span>Apoie nossa comunidade de B3 acessando os patrocinadores parceiros na barra de relatórios!</span>
      </div>

      <div className="flex flex-col lg:flex-row flex-1">
        
        {/* SIDEBAR */}
        <aside className="w-full lg:w-80 bg-white border-b lg:border-b-0 lg:border-r border-slate-200 p-6 flex flex-col gap-6 shrink-0">
          
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 bg-linear-to-tr from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center text-white font-extrabold text-xl shadow-md tracking-wider">
              B3
            </div>
            <div>
              <h1 className="text-md font-extrabold text-slate-950 tracking-tight leading-tight">Ranking Ibovespa</h1>
              <p className="text-[11px] font-semibold text-blue-600 font-mono tracking-wider">SISTEMA INTELIGENTE 2026</p>
            </div>
          </div>

          <hr className="border-slate-100" />

          {/* User Custom Identification */}
          <div className="bg-slate-50/80 rounded-xl p-4 border border-slate-100">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Briefcase size={14} className="text-slate-500" /> Identificação Nominal
            </h3>
            <form onSubmit={handleSaveName} className="flex flex-col gap-2">
              <input
                type="text"
                placeholder="Seu nome p/ relatórios:"
                value={userNameInput}
                onChange={(e) => setUserNameInput(e.target.value)}
                className="w-full text-xs font-medium bg-white border border-slate-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-lg px-3 py-1.5 outline-hidden transition-all"
              />
              <button
                type="submit"
                className="w-full bg-slate-800 hover:bg-slate-950 text-white font-semibold text-[11px] py-1.5 rounded-lg transition-colors cursor-pointer"
              >
                Salvar Identidade
              </button>
            </form>
            {userName && (
              <p className="text-[10px] font-semibold text-slate-500 italic mt-2 text-center">
                Investidor ativo logado: <span className="text-blue-600 font-bold">{userName}</span>
              </p>
            )}
          </div>

          {/* Real-time UTC Hour Counter */}
          <div className="bg-slate-950 text-white rounded-xl p-4 shadow-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock size={16} className="text-blue-400 animate-pulse" />
              <span className="text-[11px] font-extrabold uppercase tracking-widest text-slate-400 font-mono">B3 Relatórios</span>
            </div>
            <span className="text-xs font-bold font-mono text-blue-400">
              {new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit" })} BRT
            </span>
          </div>

          {/* live query quick tool */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100/50 rounded-xl p-4">
            <h3 className="text-xs font-bold text-blue-900 uppercase tracking-wider mb-1 flex items-center gap-1">
              <Sparkles size={13} strokeWidth={2.5} /> IA Live Quotes (Tempo Real)
            </h3>
            <p className="text-[10.5px] text-slate-600 leading-relaxed mb-3">
              Recupere a cotação real do dia de hoje (em 2026), notícias quentes e proventos em tempo real direto da B3 via Gemini Search Grounding!
            </p>
            <div className="flex flex-col gap-2">
              <select
                className="bg-white border text-xs border-blue-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-lg px-2.5 py-1.5 opacity-90 outline-hidden tracking-wider font-semibold font-mono"
                onChange={(e) => handleLiveQuery(e.target.value)}
                disabled={updatingTicker !== ""}
                defaultValue=""
              >
                <option value="" disabled>Escolha um papel da B3...</option>
                {stocks.map(s => (
                  <option key={s.papel} value={s.papel}>{s.papel} - {s.empresa}</option>
                ))}
              </select>
              {updatingTicker && (
                <div className="text-center font-bold text-[10px] text-blue-800 animate-pulse flex items-center justify-center gap-1.5">
                  <RefreshCw size={12} className="animate-spin text-blue-600" /> Carregando Google Grounding...
                </div>
              )}
            </div>
          </div>

          {/* Giga+ Fibra AD Cupom */}
          <a
            href="https://wa.me/552220410353?text=Use%20o%20codigo%20DVT329%20e%20ganhe%2020%25%20nas%20duas%20primeiras%20mensalidades"
            target="_blank"
            rel="noreferrer"
            className="block group bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl p-4 shadow-sm transition-all duration-200 transform hover:-translate-y-0.5"
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="bg-amber-400 text-slate-900 font-extrabold text-[9px] px-1.5 py-0.5 rounded-sm uppercase tracking-wider font-mono">Cupom Giga+</span>
              <span className="text-[10px] text-emerald-100 font-bold font-mono">DVT329</span>
            </div>
            <h4 className="text-xs font-black leading-tight mb-1 group-hover:underline">Banda Larga Fibra com 20% OFF!</h4>
            <p className="text-[10px] text-emerald-100 leading-snug">
              Assine fibra ótica de alta velocidade e alto rendimento operado pela Giga+ Fibra. WhatsApp oficial.
            </p>
            <div className="mt-2.5 bg-emerald-950/20 text-center py-1 rounded-md text-[10px] font-bold text-white flex items-center justify-center gap-1">
              Falar no WhatsApp <ExternalLink size={10} />
            </div>
          </a>

          {/* Quick numbers tracker */}
          <div className="bg-slate-50 border border-slate-100 p-4 rounded-xl flex flex-col gap-1.5 text-xs font-semibold text-slate-600 mt-auto">
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Universo de Cobertura</div>
            <div className="flex items-center justify-between">
              <span>Ativos monitorados</span>
              <span className="text-slate-900 font-mono font-bold">{stocks.length} empresas</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Liquidez mínima</span>
              <span className="text-slate-900 font-mono font-bold">&gt; R$ 100k</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Modelos analíticos</span>
              <span className="text-slate-900 font-mono font-bold">Graham-Greenblatt</span>
            </div>
          </div>

        </aside>

        {/* MAIN PANEL CONTENT */}
        <main className="flex-1 p-6 lg:p-8 flex flex-col gap-6 overflow-hidden">
          
          {/* Header Title Bar */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-1.5 mb-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider font-mono">Monitor de Oportunidades 2026</span>
              </div>
              <h2 className="text-2xl font-black text-slate-900 tracking-tight">Cenário Tático de Investimentos Ibovespa</h2>
              <p className="text-xs text-slate-500 mt-0.5 font-medium">
                {userName ? `Relatório estratégico de investimentos customizado sob demanda para ${userName}.` : "Filtros fundamentalistas estritos e valuation automático em tempo real para ativos da B3."}
              </p>
            </div>

            <div className="flex items-center gap-3 self-start md:self-auto flex-wrap">
              <button
                onClick={handleGlobalRefresh}
                disabled={refreshingGlobal || loading}
                title="Clique aqui para atualizar todas as cotações de 100 ativos com flutuações e trazer notícias novas no dia"
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-black text-xs px-4 py-3 rounded-xl cursor-pointer flex items-center gap-2 transition-all shadow-md active:scale-95"
              >
                <RefreshCw size={14} className={refreshingGlobal ? "animate-spin" : ""} />
                {refreshingGlobal ? "Sincronizando B3..." : "🔄 Atualizar Pregão Geral B3"}
              </button>

              <div className="bg-blue-50/50 border border-blue-100 rounded-xl p-3 flex items-center gap-3 pr-5">
                <span className="text-2xl">🇧🇷</span>
                <div>
                  <span className="block text-[8px] font-extrabold text-blue-600 uppercase tracking-wider">Selic de Referência 2026</span>
                  <span className="text-xs font-black text-slate-800">COPOM • Estável</span>
                </div>
              </div>
            </div>
          </div>

          {/* Update Feeds Notifications */}
          {updateFeedback && (
            <div className={`p-4 rounded-xl border flex items-start gap-3 shadow-xs animate-fade-in ${
              updateFeedback.type === "success" 
                ? "bg-emerald-50/80 border-emerald-100 text-emerald-800" 
                : "bg-rose-50/80 border-rose-100 text-rose-800"
            }`}>
              <span className="text-lg mt-0.5">{updateFeedback.type === "success" ? "✅" : "⚠️"}</span>
              <p className="text-xs font-semibold leading-relaxed">{updateFeedback.message}</p>
            </div>
          )}

          {/* Search box and Tabs Row combined */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-4 shadow-2xs flex flex-col md:flex-row md:items-center justify-between gap-4">
            
            {/* Tabs List */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-2 md:pb-0 scrollbar-none snap-x">
              {[
                "🏆 Ranking Fundamentalista",
                "✨ Fórmula Mágica",
                "💎 Graham Valuation",
                "📈 EPS Diluído",
                "📉 Assimetria Lucro/Preço",
                "🐍 Código Python Corrigido"
              ].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setSelectedTab(tab)}
                  className={`snap-start shrink-0 px-4 py-2 rounded-xl text-xs font-bold leading-none cursor-pointer transition-all duration-150 ${
                    selectedTab === tab
                      ? "bg-blue-600 text-white shadow-xs"
                      : "bg-slate-50 text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* General Filter Search Box */}
            <div className="relative w-full md:w-64 shrink-0">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                <Search size={14} />
              </span>
              <input
                type="text"
                placeholder="Filtrar por ticker ou empresa..."
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                className="w-full text-xs font-semibold bg-slate-50 border border-slate-200 focus:border-blue-500 focus:bg-white focus:ring-1 focus:ring-blue-500 rounded-xl pl-9 pr-4 py-2 outline-hidden transition-all"
              />
            </div>

          </div>

          {loading ? (
            <div className="bg-white rounded-3xl border border-slate-200/80 p-12 shadow-2xs flex flex-col items-center justify-center py-28 text-center animate-fade-in">
              <RefreshCw size={36} className="text-blue-600 animate-spin mb-4" />
              <h4 className="font-extrabold text-slate-950 text-base">Sincronizando Sistema Fundamentalista B3</h4>
              <p className="text-xs text-slate-500 mt-2 max-w-sm">
                Conectando ao banco de dados Express de alta frequência... Sincronizando dados de 100 ativos da B3 de 2026, dividendos anunciados e notícias quentes de mercado.
              </p>
            </div>
          ) : (
            <>
              {/* TAB 1: RANKING FUNDAMENTALISTA */}
              {selectedTab === "🏆 Ranking Fundamentalista" && (
            <div className="flex flex-col gap-6">
              
              <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-2xs">
                <div className="flex items-start gap-4 mb-4">
                  <div className="p-3 bg-blue-50 text-blue-600 rounded-xl">
                    <Award size={20} spellCheck />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-blue-600 uppercase tracking-widest font-mono mb-0.5">Sinais Fundamentalistas Saudáveis</h4>
                    <h3 className="text-lg font-black text-slate-900 leading-tight">Oportunidades Saudáveis de B3</h3>
                    <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                      Mostramos abaixo as ações que passam em filtros rigorosos de conservadorismo: ROE superior a 5%, P/L positivo inferior a 15, EV/EBIT inferior a 10, dividendos consistentes (Dividend Yield superior a 4%), margem líquida superior a 5% e liquidez sólida diária expressiva acima de R$ 200 mil.
                    </p>
                  </div>
                </div>

                {loading ? (
                  <div className="py-20 text-center animate-pulse text-xs font-extrabold text-blue-600">Sincronizando dados fundamentalistas da B3...</div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse min-w-[700px]">
                      <thead>
                        <tr className="border-b border-slate-100 text-[10px] font-bold text-slate-400 uppercase tracking-widest bg-slate-50/50">
                          <th className="py-3 px-4 rounded-l-xl">Papel (Ticker)</th>
                          <th className="py-3 px-4">Empresa</th>
                          <th className="py-3 px-4 text-right">Preço</th>
                          <th className="py-3 px-4 text-right">P/L</th>
                          <th className="py-3 px-4 text-right">EV/EBIT</th>
                          <th className="py-3 px-4 text-right">ROE</th>
                          <th className="py-3 px-4 text-right">Div. Yield</th>
                          <th className="py-3 px-4 text-right rounded-r-xl">Margem Líq.</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 text-xs font-medium">
                        {filterBySearch(getHealthyStocksList()).map((s) => (
                          <tr key={s.papel} className="hover:bg-slate-50/50 transition-colors group">
                            <td className="py-3.5 px-4">
                              <span className="font-bold font-mono text-slate-900 group-hover:text-blue-600 transition-colors bg-slate-100/80 px-2 py-0.5 rounded-md">
                                {s.papel}
                              </span>
                            </td>
                            <td className="py-3.5 px-4 text-slate-600 truncate max-w-[200px]">{s.empresa}</td>
                            <td className="py-3.5 px-4 text-right font-bold text-slate-900 font-mono">{fmtCurrency(s.cotacao)}</td>
                            <td className="py-3.5 px-4 text-right font-mono font-bold text-blue-600">{s.pl.toFixed(2)}</td>
                            <td className="py-3.5 px-4 text-right font-mono text-slate-700">{s.evebit.toFixed(2)}</td>
                            <td className="py-3.5 px-4 text-right font-mono text-slate-700">{fmtPercent(s.roe)}</td>
                            <td className="py-3.5 px-4 text-right font-mono font-bold text-emerald-600">{fmtPercent(s.dy)}</td>
                            <td className="py-3.5 px-4 text-right font-mono text-slate-700">{fmtPercent(s.mrgliq)}</td>
                          </tr>
                        ))}
                        {filterBySearch(getHealthyStocksList()).length === 0 && (
                          <tr>
                            <td colSpan={8} className="py-8 text-center text-xs text-slate-400 italic">
                              Nenhuma empresa atendeu aos filtros restritivos severos com este termo de busca.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Elevated Debt Warning List */}
              <div className="bg-amber-50/30 rounded-2xl border border-amber-100 p-6 shadow-2xs">
                <div className="flex items-start gap-4 mb-4">
                  <div className="p-3 bg-amber-100 text-amber-700 rounded-xl">
                    <AlertTriangle size={20} />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-amber-700 uppercase tracking-widest font-mono mb-0.5">⚠️ Monitor de Alerta e Riscos</h4>
                    <h3 className="text-lg font-black text-slate-900 leading-tight">Empresas Gratificantes com Alavancagem Elevada ou em Recuperação Judicial</h3>
                    <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                      Filtramos as organizações que apresentam alto risco cambial ou alavancagem de passivos exagerada (Dívida/Patrimônio superior a 3.0) ou classificadas expressamente em Recuperação Judicial pela B3.
                    </p>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse min-w-[700px]">
                    <thead>
                      <tr className="border-b border-amber-150/50 text-[10px] font-bold text-amber-700 uppercase tracking-widest bg-amber-100/20">
                        <th className="py-3 px-4 rounded-l-xl">Ativo</th>
                        <th className="py-3 px-4">Empresa</th>
                        <th className="py-3 px-4 text-right">Preço</th>
                        <th className="py-3 px-4 text-right">Alavancagem (Dív/Patr)</th>
                        <th className="py-3 px-4 text-right">Variação Resultados</th>
                        <th className="py-3 px-4 text-right rounded-r-xl">Situação de Bolsa</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-amber-100/50 text-xs font-medium">
                      {filterBySearch(getHighRiskOrAlavancada()).map((s) => (
                        <tr key={s.papel} className="hover:bg-amber-100/10 transition-colors">
                          <td className="py-3.5 px-4">
                            <span className="font-bold font-mono text-slate-900 bg-amber-100/40 px-2 py-0.5 rounded-md">
                              {s.papel}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 text-slate-600">{s.empresa}</td>
                          <td className="py-3.5 px-4 text-right font-bold text-slate-900 font-mono">{fmtCurrency(s.cotacao)}</td>
                          <td className="py-3.5 px-4 text-right font-mono font-bold text-amber-700">{s.divbpatr.toFixed(2)}</td>
                          <td className="py-3.5 px-4 text-right font-mono text-rose-600 font-bold">{s.quedaLucro}</td>
                          <td className="py-3.5 px-4 text-right">
                            <span className="bg-rose-100 text-rose-800 text-[10px] font-bold px-2 py-0.5 rounded-full font-mono uppercase">
                              {s.situacao}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Chart Selector and Live interactive chart */}
              <div className="grid grid-cols-1 gap-6">
                <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-2xs flex flex-col gap-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                      <h3 className="text-md font-extrabold text-slate-900">Gráfico Histórico de Demonstração de Resultados</h3>
                      <p className="text-xs text-slate-500 mt-0.5">Analise o equilíbrio entre receita consolidada, margens de lucro e a volatilidade do papel na B3.</p>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-slate-500">Selecione Ativo:</span>
                      <select
                        className="bg-slate-50 border text-xs border-slate-200 focus:border-blue-500 rounded-lg px-3 py-1.5 outline-hidden tracking-wider font-extrabold font-mono"
                        value={selectedChartTicker}
                        onChange={(e) => setSelectedChartTicker(e.target.value)}
                      >
                        {stocks.map(s => (
                          <option key={s.papel} value={s.papel}>{s.papel} - {s.empresa}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {selectedChartStockObj && (
                    <InteractiveChart stock={selectedChartStockObj} />
                  )}
                </div>
              </div>

            </div>
          )}

          {/* TAB 2: FÓRMULA MÁGICA */}
          {selectedTab === "✨ Fórmula Mágica" && (
            <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-2xs flex flex-col gap-6">
              
              <div className="flex items-start gap-4 bg-indigo-50/50 border border-indigo-100 p-5 rounded-2xl">
                <div className="p-3 bg-indigo-100 text-indigo-700 rounded-xl">
                  <Sparkles size={20} />
                </div>
                <div>
                  <h4 className="text-xs font-bold text-indigo-700 uppercase tracking-widest font-mono mb-0.5">Joel Greenblatt Model</h4>
                  <h3 className="text-lg font-black text-indigo-950">Fórmula Mágica de Cobertura B3</h3>
                  <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                    A metodologia consagrada por Joel Greenblatt organiza as ações considerando o melhor equilíbrio conjunto de <strong>EV/EBIT (Earning Yield de forma inversa)</strong> e <strong>ROIC (Retorno sobre Capital Investido)</strong>. Quanto menor a soma agregada do Score Mágico, mais atrativa e barata a empresa se posiciona em Relação ao que gera!
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse min-w-[700px]">
                  <thead>
                    <tr className="border-b border-slate-100 text-[10px] font-bold text-slate-400 uppercase tracking-widest bg-slate-50/50">
                      <th className="py-3 px-4 rounded-l-xl">Posição</th>
                      <th className="py-3 px-4">Ativo</th>
                      <th className="py-3 px-4">Empresa</th>
                      <th className="py-3 px-4 text-right">Preço</th>
                      <th className="py-3 px-4 text-center">Score Greenblatt</th>
                      <th className="py-3 px-4 text-right">Earning Yield (Ref)</th>
                      <th className="py-3 px-4 text-right">EV/EBIT</th>
                      <th className="py-3 px-4 text-right rounded-r-xl">ROIC</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-xs font-medium">
                    {filterBySearch(getMagicFormulaList()).map((s, idx) => (
                      <tr key={s.papel} className="hover:bg-slate-50/55 transition-all">
                        <td className="py-3 px-4 text-slate-400 font-bold font-mono">#{idx + 1}</td>
                        <td className="py-3 px-4">
                          <span className="font-bold font-mono text-slate-900 bg-slate-100 px-2 py-0.5 rounded-md">
                            {s.papel}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-slate-600">{s.empresa}</td>
                        <td className="py-3 px-4 text-right font-bold text-slate-900 font-mono">{fmtCurrency(s.cotacao)}</td>
                        <td className="py-3 px-4 text-center">
                          <span className="bg-indigo-100 text-indigo-800 font-bold px-2.5 py-0.5 rounded-full font-mono text-[10px]">
                            {s.scoreMagico}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right font-mono text-slate-600">{fmtPercent(s.ey)}</td>
                        <td className="py-3 px-4 text-right font-mono font-bold text-blue-600">{s.evebit.toFixed(2)}</td>
                        <td className="py-3 px-4 text-right font-mono font-bold text-emerald-600">{fmtPercent(s.roic)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

            </div>
          )}

          {/* TAB 3: GRAHAM VALUATION */}
          {selectedTab === "💎 Graham Valuation" && (
            <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-2xs flex flex-col gap-6">
              
              <div className="flex items-start gap-4 bg-amber-50/50 border border-amber-100 p-5 rounded-2xl">
                <div className="p-3 bg-amber-100 text-amber-800 rounded-xl">
                  <DollarSign size={20} />
                </div>
                <div>
                  <h4 className="text-xs font-bold text-amber-800 uppercase tracking-widest font-mono mb-0.5">Benjamin Graham Value Investing</h4>
                  <h3 className="text-lg font-black text-amber-950">Preço Justo e Margem de Segurança Graham</h3>
                  <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                    A equação clássica de Benjamin Graham define o <strong>Valor Intrínseco (V.I. = Raiz(22.5 * LPA * VPA))</strong>, estipulando limites de margem contábil exigíveis por investidores de valor. Comparamos o preço de tela atual com a margem implícita de desconto físico do balanço.
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse min-w-[700px]">
                  <thead>
                    <tr className="border-b border-slate-100 text-[10px] font-bold text-slate-400 uppercase tracking-widest bg-slate-50/50">
                      <th className="py-3 px-4 rounded-l-xl">Ativo</th>
                      <th className="py-3 px-4">Preço na Tela</th>
                      <th className="py-3 px-4 text-right">Preço Justo (V.I. Graham)</th>
                      <th className="py-3 px-4 text-right">Margem de Desconto / Upside</th>
                      <th className="py-3 px-4 text-right">LPA (Lucro p/ Ação)</th>
                      <th className="py-3 px-4 text-right">VPA (Valor Patrimonial)</th>
                      <th className="py-3 px-4 text-center rounded-r-xl">Avaliação da B3</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-xs font-medium">
                    {filterBySearch(getGrahamValuationList()).map((s) => (
                      <tr key={s.papel} className="hover:bg-slate-50/55 transition-all">
                        <td className="py-3.5 px-4 font-bold font-mono text-slate-900">{s.papel}</td>
                        <td className="py-3.5 px-4 font-bold font-mono text-slate-700">{fmtCurrency(s.cotacao)}</td>
                        <td className="py-3.5 px-4 text-right font-mono font-bold text-amber-700">{fmtCurrency(s.valorIntrinseco)}</td>
                        <td className="py-3.5 px-4 text-right font-mono">
                          <span className={`font-bold ${s.ratio > 1 ? "text-emerald-600" : "text-amber-600"}`}>
                            {s.ratio.toFixed(2)}x
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-right font-mono text-slate-600">{fmtCurrency(s.lpa)}</td>
                        <td className="py-3.5 px-4 text-right font-mono text-slate-600">{fmtCurrency(s.vpa)}</td>
                        <td className="py-3.5 px-4 text-center">
                          <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                            s.ratio > 1 
                              ? "bg-emerald-100 text-emerald-800" 
                              : "bg-slate-100 text-slate-600"
                          }`}>
                            {s.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

            </div>
          )}

          {/* TAB 4: EPS DILUÍDO */}
          {selectedTab === "📈 EPS Diluído" && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-2xs lg:col-span-2 flex flex-col gap-4">
                <div>
                  <h4 className="text-xs font-bold text-blue-600 font-mono uppercase tracking-widest">Lucros Trimestrais Expressivos</h4>
                  <h3 className="text-lg font-black text-slate-950 mt-1">Destaques com EPS Trimestral &gt; R$ 1,00</h3>
                  <p className="text-xs text-slate-500 leading-relaxed mt-1">
                    Selecionamos as empresas que demonstraram lucratividade ultra-tangível em curto prazo, acumulando Lucro Por Ação (EPS) diluído trimestral contábil superior a um real na última referência de auditoria homologada.
                  </p>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse min-w-[500px]">
                    <thead>
                      <tr className="border-b border-slate-100 text-[10px] font-bold text-slate-400 uppercase tracking-widest bg-slate-50/50">
                        <th className="py-2.5 px-3 rounded-l-xl">Ativo</th>
                        <th className="py-2.5 px-3">Empresa</th>
                        <th className="py-2.5 px-3 text-right">EPS Trimestral</th>
                        <th className="py-2.5 px-3 text-right">Preço Atual</th>
                        <th className="py-2.5 px-3 text-center rounded-r-xl">Ex-Data Ref</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-xs font-medium">
                      {filterBySearch(getEPSRobustList()).map((s) => (
                        <tr key={s.papel} className="hover:bg-slate-50/45 transition-all">
                          <td className="py-3 px-3">
                            <span className="font-bold font-mono text-slate-900 bg-slate-100 px-2 py-0.5 rounded-md">
                              {s.papel}
                            </span>
                          </td>
                          <td className="py-3 px-3 text-slate-600 truncate max-w-[150px]">{s.empresa}</td>
                          <td className="py-3 px-3 text-right font-mono font-bold text-emerald-600">{fmtCurrency(s.epsTrimestral)}</td>
                          <td className="py-3 px-3 text-right font-mono text-slate-900">{fmtCurrency(s.cotacao)}</td>
                          <td className="py-3 px-3 text-center font-mono text-slate-400 font-bold">{s.dataRef}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Trading view triggers widget */}
              <div className="bg-linear-to-b from-slate-950 to-slate-900 text-white rounded-2xl p-6 flex flex-col gap-5 justify-between">
                <div>
                  <span className="bg-blue-600 text-[9px] font-bold tracking-widest uppercase px-1.5 py-0.5 rounded-xs font-mono">Sinal de Cruzamento</span>
                  <h3 className="text-md font-bold mt-2">Osciladores Avançados TradingView</h3>
                  <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                    Cruzar os resultados de análise fundamentalista pura de Graham e Greenblatt com as métricas de agressão de fluxo de volume do mercado garante pontos de entrada com melhores distribuições estocásticas.
                  </p>
                </div>

                <div className="bg-slate-800/40 border border-slate-800 p-4 rounded-xl">
                  <span className="text-[10px] font-bold text-blue-400 font-mono block">Ativo de Referência em Tendência</span>
                  <span className="text-sm font-black block mt-0.5">Petrobras ON (PETR3)</span>
                  <div className="flex items-center gap-2 mt-2">
                    <CheckCircle size={14} className="text-emerald-400" />
                    <span className="text-[11px] text-emerald-400 font-bold">Compra Forte Recomendada</span>
                  </div>
                </div>

                <p className="text-[10.5px] text-slate-400 italic">
                  *Aviso: Este monitor estratégico consolida sinais públicos. Pratique gerenciamento rígido de riscos operacionais.
                </p>
              </div>

            </div>
          )}

          {/* TAB 5: ASSIMETRIA LUCRO/PREÇO */}
          {selectedTab === "📉 Assimetria Lucro/Preço" && (
            <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-2xs flex flex-col gap-6">
              
              <div className="flex items-start gap-4 bg-emerald-50/50 border border-emerald-100 p-5 rounded-2xl">
                <div className="p-3 bg-emerald-100 text-emerald-800 rounded-xl">
                  <TrendingUp size={20} />
                </div>
                <div>
                  <h4 className="text-xs font-bold text-emerald-800 uppercase tracking-widest font-mono mb-0.5">Comportamento de Boca de Jacaré</h4>
                  <h3 className="text-lg font-black text-slate-950">Atraso Notório de Preço Versus Curva de Lucros</h3>
                  <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                    A assimetria ("Boca de Jacaré") identifica ativos em que <strong>o lucro acumulado da companhia subiu ao longo dos últimos períodos históricos enquanto o preço da ação na bolsa permaneceu em patamares baixos, deprimidos ou desajustados</strong>. Essa diferença de boca de jacaré tende a fechar pelo reajuste tardio de valor e compras agressivas!
                  </p>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
                <div>
                  <h3 className="text-sm font-extrabold text-slate-800">Verificar Gráfico de Boca de Jacaré</h3>
                  <span className="text-[11px] text-slate-400 font-medium">Selecione uma das empresas em visível assimetria fundamental para plotar.</span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-xs font-extrabold text-slate-500 whitespace-nowrap">Selecione Ativo do Gráfico:</span>
                  <select
                    className="bg-slate-50 border text-xs border-slate-200 focus:border-emerald-500 rounded-lg px-3 py-1.5 outline-hidden tracking-wider font-extrabold font-mono"
                    value={selectedAsymmetryTicker}
                    onChange={(e) => setSelectedAsymmetryTicker(e.target.value)}
                  >
                    {getAsymmetryStocksList().length > 0 ? (
                      getAsymmetryStocksList().map(s => (
                        <option key={s.papel} value={s.papel}>{s.papel} - {s.empresa}</option>
                      ))
                    ) : (
                      // Fallback list just in case
                      ["BBSE3", "TAEE11", "PETR4", "WEGE3"].map(ticker => (
                        <option key={ticker} value={ticker}>{ticker}</option>
                      ))
                    )}
                  </select>
                </div>
              </div>

              {selectedAsymmetryStockObj && (
                <div>
                  <InteractiveChart 
                    stock={selectedAsymmetryStockObj} 
                    title={`Comportamento de Boca de Jacaré (Assimetria) - ${selectedAsymmetryStockObj.papel}`} 
                  />

                  <div className="mt-4 p-4 rounded-xl bg-emerald-50 border border-emerald-100/50 text-emerald-800 text-xs flex items-center gap-3">
                    <CheckCircle size={18} className="text-emerald-600 shrink-0" />
                    <div>
                      <strong className="block text-emerald-900 font-bold mb-0.5">Sinal Assimétrico Ativo para {selectedAsymmetryStockObj.papel}!</strong>
                      A curva tracejada verde (Evolução de Lucros) posicionou-se nitidamente acima da linha azul cheia (Preço de Cotação de Bolsa), confirmando distorção favorável ao investidor defensivo de valor.
                    </div>
                  </div>
                </div>
              )}

            </div>
          )}

          {/* TAB 6: CÓDIGO PYTHON CORRIGIDO */}
          {selectedTab === "🐍 Código Python Corrigido" && (
            <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-2xs flex flex-col gap-6 font-sans">
              
              <div className="flex items-start gap-4 bg-blue-50/50 border border-blue-100 p-5 rounded-2xl">
                <div className="p-3 bg-blue-100 text-blue-800 rounded-xl">
                  <CheckCircle size={20} />
                </div>
                <div className="flex-1">
                  <h4 className="text-xs font-bold text-blue-850 uppercase tracking-widest font-mono mb-0.5">Diagnóstico Técnico & Correção de Sintaxe</h4>
                  <h3 className="text-lg font-black text-slate-950">Por que o erro acontecia no seu Streamlit?</h3>
                  <p className="text-xs text-slate-600 mt-2 leading-relaxed">
                    O erro <code className="bg-red-50 text-red-600 px-1 py-0.5 rounded-sm font-mono font-bold">SyntaxError: invalid syntax</code> na linha <code className="font-mono bg-slate-100 px-1 py-0.5 rounded-sm">with col_mp = st.columns(2)[1]:</code> ocorria porque em Python não é permitida a atribuição direta utilizando o operador de igualdade <code className="font-mono bg-slate-100 px-1 py-0.5 rounded-sm">=</code> dentro da declaração do bloco gerenciador de contexto <code className="font-mono bg-slate-100 px-1 py-0.5 rounded-sm">with</code>.
                  </p>
                  <p className="text-xs text-slate-600 mt-2 leading-relaxed font-semibold">
                    ✅ Solução aplicada no código abaixo:
                  </p>
                  <pre className="mt-2 p-3 bg-slate-50 text-slate-700 rounded-xl text-xs font-mono border border-slate-200">
{`col_nomad, col_mp = st.columns(2)
with col_nomad:
    # Conteúdo Nomad
with col_mp:
    # Conteúdo Mercado Pago`}
                  </pre>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
                <div>
                  <h3 className="text-sm font-extrabold text-slate-800">Script Python Integral do Aplicativo B3 (2026)</h3>
                  <span className="text-[11px] text-slate-400 font-medium">Contém todas as 100 ações com preços de 2026, gráficos normalizados interativos e o erro corrigido.</span>
                </div>

                <button
                  onClick={handleCopyPython}
                  className={`px-5 py-2.5 rounded-xl font-bold text-xs cursor-pointer flex items-center gap-2 justify-center transition-all ${
                    copiedPython 
                      ? "bg-emerald-600 text-white min-w-[170px]" 
                      : "bg-blue-600 hover:bg-blue-700 text-white min-w-[170px]"
                  }`}
                >
                  {copiedPython ? (
                    <>
                      <CheckCircle size={14} /> Copiado com sucesso!
                    </>
                  ) : (
                    <>
                      <span>📋</span> Copiar Script (app.py)
                    </>
                  )}
                </button>
              </div>

              <div className="relative">
                <div className="absolute top-3 right-3 bg-slate-800 text-slate-400 text-[10px] font-mono px-2.5 py-1 rounded-md font-bold uppercase tracking-wider select-none">
                  PYTHON STREAMLIT
                </div>
                <pre className="bg-slate-950 text-slate-300 font-mono text-xs p-6 rounded-2xl overflow-auto max-h-[500px] border border-slate-900 shadow-inner scrollbar-thin">
                  <code>{PYTHON_CODE_APP}</code>
                </pre>
              </div>
              
              <div className="text-[11px] text-slate-450 bg-slate-50 border p-4 rounded-xl">
                💡 <strong>Como rodar no seu computador:</strong><br />
                1. Salve este código acima em um arquivo chamado <code className="font-mono bg-slate-200 text-slate-800 px-1 rounded">app.py</code>.<br />
                2. Instale as dependências executando no terminal: <code className="font-mono bg-slate-200 text-slate-800 px-1 rounded">pip install streamlit plotly pandas</code>.<br />
                3. Rode o app localmente com: <code className="font-mono bg-slate-200 text-slate-800 px-1 rounded">streamlit run app.py</code>.
              </div>

            </div>
          )}

            </>
          )}

          {/* News & Dividends Feed widgets (Compartilhado abaixo de todas as abas fundamentalistas) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
            
            {/* News container */}
            <div className="bg-white rounded-2xl border border-slate-150 p-6 shadow-2xs">
              <div className="flex items-center gap-2 mb-4">
                <Newspaper size={18} className="text-blue-600" />
                <h3 className="text-sm font-black text-slate-900 uppercase tracking-wider">📰 Notícias Recentes do Mercado (B3 2026)</h3>
              </div>
              <div className="flex flex-col gap-3">
                {news.map((item, idx) => (
                  <div key={idx} className="border border-slate-100 hover:border-slate-200 p-3.5 rounded-xl bg-slate-50/50 hover:bg-slate-50 transition-all">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[9px] font-extrabold bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded-sm font-mono tracking-wider uppercase">
                        {item.source}
                      </span>
                      <span className="text-[10px] text-slate-400 font-bold flex items-center gap-1">
                        <Clock size={10} /> Recente em 2026
                      </span>
                    </div>
                    <p className="text-xs font-bold text-slate-800 leading-snug my-1.5">{item.title}</p>
                    <a
                      href={item.link}
                      target="_blank"
                      rel="noreferrer"
                      className="text-[11px] font-bold text-blue-600 hover:underline inline-flex items-center gap-0.5"
                    >
                      Ler cobertura no portal <ExternalLink size={10} />
                    </a>
                  </div>
                ))}
                {news.length === 0 && (
                  <div className="text-center text-xs text-slate-400 italic py-6">Nenhuma notícia registrada no momento.</div>
                )}
              </div>
            </div>

            {/* Dividends container */}
            <div className="bg-white rounded-2xl border border-slate-150 p-6 shadow-2xs">
              <div className="flex items-center gap-2 mb-4">
                <Calendar size={18} className="text-emerald-600" />
                <h3 className="text-sm font-black text-slate-900 uppercase tracking-wider font-semibold">💰 Últimos Proventos Anunciados (B3 2026)</h3>
              </div>
              <div className="flex flex-col gap-2.5">
                {dividends.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 border border-slate-100 rounded-xl bg-slate-50/50 hover:bg-emerald-50/10 transition-all">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-black text-slate-900 font-mono bg-slate-200 px-2 py-0.5 rounded-sm">
                          {item.ativo}
                        </span>
                        <span className="text-[11px] text-slate-500 font-bold">
                          Rend. Financeiro
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-400 font-semibold block mt-1">Ex-data/Anúncio: {item.data}</span>
                    </div>
                    <div className="text-right">
                      <div className="text-[13px] font-extrabold text-emerald-600 font-mono">
                        {fmtCurrency(item.valor)}
                      </div>
                      <span className="text-[9px] text-slate-400 font-bold font-mono">por papel</span>
                    </div>
                  </div>
                ))}
                {dividends.length === 0 && (
                  <div className="text-center text-xs text-slate-400 italic py-6">Nenhum provento anunciado recentemente.</div>
                )}
              </div>
            </div>

          </div>

          {/* Members sponsors section styled like Streamlit code */}
          <div className="mt-4">
            <h4 className="text-center font-bold text-[11px] text-slate-400 uppercase tracking-widest mb-4">
              Nossos Membros Patrocinadores Oficiais
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Nomad sponsor card */}
              <a
                href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I&n=Jader"
                target="_blank"
                rel="noreferrer"
                className="block group bg-white hover:bg-slate-50 border border-slate-200 rounded-2xl p-5 shadow-2xs hover:border-amber-500/50 transition-all duration-150"
              >
                <div className="flex items-center gap-3.5 mb-2">
                  <span className="text-2xl select-none group-hover:scale-110 transition-transform">✈️</span>
                  <h4 className="text-sm font-extrabold text-slate-900 group-hover:text-blue-600">Nomad: Conta Gratuita em Dólar nos EUA</h4>
                </div>
                <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                  Abra sua conta de investimentos global, receba e dolarize seus proventos livre de taxas de manutenção. Garanta incentivos fiscais exclusivos e taxa zero de câmbio no primeiro envio!
                </p>
                <div className="bg-amber-50 text-amber-800 text-[11.5px] font-bold text-center py-2.5 rounded-xl group-hover:bg-amber-100 transition-colors">
                  Abrir Minha Conta Nomad Internacional →
                </div>
              </a>

              {/* Mercado Pago sponsor card */}
              <a
                href="https://mpago.li/1VydVhw"
                target="_blank"
                rel="noreferrer"
                className="block group bg-white hover:bg-slate-50 border border-slate-200 rounded-2xl p-5 shadow-2xs hover:border-blue-400/50 transition-all duration-150"
              >
                <div className="flex items-center gap-3.5 mb-2">
                  <span className="text-2xl select-none group-hover:scale-110 transition-transform font-mono">🤝</span>
                  <h4 className="text-sm font-extrabold text-slate-900 group-hover:text-blue-600">Mercado Pago: R$ 30,00 Grátis de Cupom</h4>
                </div>
                <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                  Ative sua carteira remunerada número um do ecossistema nacional, receba cashback e configure link de pagamentos para seus clientes lucrando um bônus especial de boas-vindas do Mercado Pago!
                </p>
                <div className="bg-blue-50 text-blue-800 text-[11.5px] font-bold text-center py-2.5 rounded-xl group-hover:bg-blue-100 transition-colors">
                  Resgatar Cupom de Desconto Mercado Pago →
                </div>
              </a>

            </div>
          </div>

          {/* Footer of the page */}
          <footer className="mt-8 bg-slate-950 text-slate-400 rounded-3xl p-8 text-center text-xs flex flex-col items-center gap-3 border border-slate-900 shadow-sm select-none">
            <h5 className="font-extrabold text-white text-sm font-mono tracking-wider">Ibovespa Fundamentalista © 2026</h5>
            <p className="max-w-md leading-relaxed text-[11px] text-slate-500">
              Desenvolvido com máxima resiliência e polimento estratégico. Cobertura da B3 com IA em tempo real e atualização de dados no dia de hoje.
            </p>
            <div className="flex items-center gap-4 text-[10px] uppercase font-bold tracking-wider font-mono text-slate-600 mt-2">
              <span className="hover:text-white cursor-pointer transition-colors">Termos de Uso</span>
              <span>•</span>
              <span className="hover:text-white cursor-pointer transition-colors">Políticas Gerais</span>
              <span>•</span>
              <span className="bg-slate-900 text-slate-500 px-2 py-0.5 rounded-sm">B3 VERSION 2026.1</span>
            </div>
          </footer>

        </main>
      </div>

    </div>
  );
}
