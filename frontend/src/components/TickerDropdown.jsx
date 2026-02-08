/**
 * Dropdown para seleccionar tickers.
 * Permite al usuario elegir que ticker visualizar en el grafico.
 */

export default function TickerDropdown({ tickers, selected, onChange }) {
  return (
    <select
      value={selected || ''}
      onChange={(e) => onChange(e.target.value)}
      className="bg-gray-800 border border-gray-600 text-white rounded-lg px-4 py-2 
                 focus:ring-2 focus:ring-cyan-500 focus:border-transparent
                 cursor-pointer hover:bg-gray-700 transition"
    >
      {tickers?.map((ticker) => (
        <option key={ticker} value={ticker}>
          {ticker}
        </option>
      ))}
    </select>
  )
}
