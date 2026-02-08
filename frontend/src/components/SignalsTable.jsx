/**
 * Tabla de senales de todos los tickers.
 * Permite ver y seleccionar tickers para ver detalles.
 */

export default function SignalsTable({ signals, onSelect, selected }) {
  if (!signals?.length) {
    return <p className="text-gray-400">No hay senales disponibles</p>
  }

  const signalColors = {
    STRONG_BUY: 'bg-green-500/20 text-green-400',
    BUY: 'bg-green-500/10 text-green-300',
    SELL: 'bg-red-500/20 text-red-400',
    HOLD: 'bg-yellow-500/10 text-yellow-400'
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="text-left py-3 px-4 text-gray-400 font-medium">Ticker</th>
            <th className="text-left py-3 px-4 text-gray-400 font-medium">Precio</th>
            <th className="text-left py-3 px-4 text-gray-400 font-medium">Probabilidad</th>
            <th className="text-left py-3 px-4 text-gray-400 font-medium">Senal</th>
            <th className="text-left py-3 px-4 text-gray-400 font-medium">Fecha</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((signal) => (
            <tr 
              key={signal.symbol}
              onClick={() => onSelect(signal.symbol)}
              className={`border-b border-gray-800 cursor-pointer transition hover:bg-white/5 ${
                selected === signal.symbol ? 'bg-cyan-900/20' : ''
              }`}
            >
              <td className="py-3 px-4 font-semibold">{signal.symbol}</td>
              <td className="py-3 px-4">${signal.price}</td>
              <td className="py-3 px-4">
                <div className="flex items-center gap-2">
                  <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${signal.probability > 0.6 ? 'bg-green-500' : signal.probability < 0.4 ? 'bg-red-500' : 'bg-yellow-500'}`}
                      style={{ width: `${signal.probability * 100}%` }}
                    />
                  </div>
                  <span className="text-sm">{(signal.probability * 100).toFixed(1)}%</span>
                </div>
              </td>
              <td className="py-3 px-4">
                <span className={`px-2 py-1 rounded text-xs font-bold ${signalColors[signal.signal]}`}>
                  {signal.signal}
                </span>
              </td>
              <td className="py-3 px-4 text-gray-400 text-sm">{signal.date}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
