import { useState, useEffect } from 'react'
import SignalCard from './components/SignalCard'
import TickerDropdown from './components/TickerDropdown'
import PriceChart from './components/PriceChart'
import SignalsTable from './components/SignalsTable'

function App() {
  const [data, setData] = useState(null)
  const [selectedTicker, setSelectedTicker] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchDashboard()
  }, [])

  const fetchDashboard = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/dashboard')
      if (!response.ok) throw new Error('Error cargando datos')
      const result = await response.json()
      setData(result)
      if (result.tickers?.length > 0) {
        setSelectedTicker(result.tickers[0])
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runPipeline = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch('/api/run-pipeline', { method: 'POST' })
      if (!response.ok) throw new Error('Error ejecutando pipeline')
      await fetchDashboard()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-white text-xl">Cargando...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen text-white p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-green-400 bg-clip-text text-transparent">
              Trading Dashboard
            </h1>
            <p className="text-gray-400 mt-1">MVP Academico - Master Data Science</p>
          </div>
          <button
            onClick={runPipeline}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg font-medium transition"
          >
            Ejecutar Pipeline
          </button>
        </div>
      </header>

      {error && (
        <div className="mb-6 p-4 bg-red-900/50 border border-red-500 rounded-lg">
          <p className="text-red-300">{error}</p>
          <button onClick={runPipeline} className="mt-2 text-sm underline">
            Ejecutar pipeline para generar datos
          </button>
        </div>
      )}

      {data && (
        <>
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <SignalCard
              title="Mejor Oportunidad"
              ticker={data.best_opportunity?.symbol || 'N/A'}
              signal={data.best_opportunity?.signal || 'HOLD'}
              probability={data.best_opportunity?.probability || 0}
              price={data.best_opportunity?.price || 0}
              highlight
            />
            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h3 className="text-gray-400 text-sm">Total Tickers</h3>
              <p className="text-3xl font-bold mt-2">{data.total_tickers}</p>
            </div>
            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h3 className="text-gray-400 text-sm">Modelo</h3>
              <p className="text-xl font-bold mt-2">{data.model_info?.name}</p>
              <p className="text-gray-400 text-sm">AUC: {data.model_info?.auc?.toFixed(4)}</p>
            </div>
          </div>

          {/* Ticker Dropdown + Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <div className="lg:col-span-2 bg-white/5 rounded-xl p-6 border border-white/10">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">Grafico de Precios</h2>
                <TickerDropdown
                  tickers={data.tickers}
                  selected={selectedTicker}
                  onChange={setSelectedTicker}
                />
              </div>
              {selectedTicker && data.historics?.[selectedTicker] && (
                <PriceChart
                  data={data.historics[selectedTicker]}
                  symbol={selectedTicker}
                />
              )}
            </div>
            
            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h2 className="text-xl font-semibold mb-4">Detalle: {selectedTicker}</h2>
              {selectedTicker && data.signals && (
                <TickerDetail
                  signal={data.signals.find(s => s.symbol === selectedTicker)}
                />
              )}
            </div>
          </div>

          {/* Signals Table */}
          <div className="bg-white/5 rounded-xl p-6 border border-white/10">
            <h2 className="text-xl font-semibold mb-4">Todas las Senales</h2>
            <SignalsTable
              signals={data.signals}
              onSelect={setSelectedTicker}
              selected={selectedTicker}
            />
          </div>
        </>
      )}
    </div>
  )
}

function TickerDetail({ signal }) {
  if (!signal) return <p className="text-gray-400">Selecciona un ticker</p>

  const signalColors = {
    STRONG_BUY: 'text-green-400 bg-green-900/30',
    BUY: 'text-green-300 bg-green-900/20',
    SELL: 'text-red-400 bg-red-900/30',
    HOLD: 'text-yellow-400 bg-yellow-900/20'
  }

  return (
    <div className="space-y-4">
      <div>
        <p className="text-gray-400 text-sm">Precio Actual</p>
        <p className="text-2xl font-bold">${signal.price}</p>
      </div>
      <div>
        <p className="text-gray-400 text-sm">Probabilidad</p>
        <p className="text-xl font-semibold">{(signal.probability * 100).toFixed(1)}%</p>
      </div>
      <div>
        <p className="text-gray-400 text-sm">Senal</p>
        <span className={`px-3 py-1 rounded-full text-sm font-bold ${signalColors[signal.signal]}`}>
          {signal.signal}
        </span>
      </div>
      <div>
        <p className="text-gray-400 text-sm">SMA 50</p>
        <p className="text-lg">${signal.sma_50}</p>
      </div>
      <div>
        <p className="text-gray-400 text-sm">Fecha</p>
        <p className="text-sm">{signal.date}</p>
      </div>
    </div>
  )
}

export default App
