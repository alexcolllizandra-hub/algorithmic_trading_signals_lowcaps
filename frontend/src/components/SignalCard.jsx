/**
 * Tarjeta de senal destacada.
 * Muestra la mejor oportunidad de trading.
 */

export default function SignalCard({ title, ticker, signal, probability, price, highlight }) {
  const signalColors = {
    STRONG_BUY: { bg: 'from-green-600 to-green-800', text: 'text-green-100' },
    BUY: { bg: 'from-green-500 to-green-700', text: 'text-green-100' },
    SELL: { bg: 'from-red-600 to-red-800', text: 'text-red-100' },
    HOLD: { bg: 'from-yellow-600 to-yellow-800', text: 'text-yellow-100' }
  }

  const colors = signalColors[signal] || signalColors.HOLD

  return (
    <div className={`rounded-xl p-6 border border-white/10 ${
      highlight 
        ? `bg-gradient-to-br ${colors.bg}` 
        : 'bg-white/5'
    }`}>
      <h3 className={`text-sm ${highlight ? 'text-white/80' : 'text-gray-400'}`}>
        {title}
      </h3>
      <p className="text-3xl font-bold mt-2">{ticker}</p>
      <div className="mt-4 flex justify-between items-end">
        <div>
          <p className={`text-sm ${highlight ? 'text-white/70' : 'text-gray-400'}`}>
            Precio
          </p>
          <p className="text-xl font-semibold">${price}</p>
        </div>
        <div className="text-right">
          <p className={`text-sm ${highlight ? 'text-white/70' : 'text-gray-400'}`}>
            Probabilidad
          </p>
          <p className="text-xl font-semibold">{(probability * 100).toFixed(1)}%</p>
        </div>
      </div>
      <div className="mt-4">
        <span className={`px-3 py-1 rounded-full text-sm font-bold ${
          highlight 
            ? 'bg-white/20 text-white' 
            : `${colors.text} bg-white/10`
        }`}>
          {signal}
        </span>
      </div>
    </div>
  )
}
