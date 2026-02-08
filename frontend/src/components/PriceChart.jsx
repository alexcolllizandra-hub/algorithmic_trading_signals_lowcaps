/**
 * Grafico de precios usando Recharts.
 * Muestra el historico de precios de un ticker.
 */

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts'

export default function PriceChart({ data, symbol }) {
  if (!data?.dates || !data?.prices) {
    return <p className="text-gray-400">No hay datos disponibles</p>
  }

  // Preparar datos para Recharts
  const chartData = data.dates.map((date, i) => ({
    date: date.substring(0, 10),
    price: data.prices[i],
    volume: data.volumes?.[i] || 0
  }))

  // Tomar ultimos 90 dias
  const recentData = chartData.slice(-90)

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={recentData}>
          <defs>
            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey="date" 
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickFormatter={(value) => value.substring(5)}
          />
          <YAxis 
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            domain={['auto', 'auto']}
            tickFormatter={(value) => `$${value.toFixed(2)}`}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1f2937', 
              border: '1px solid #374151',
              borderRadius: '8px'
            }}
            labelStyle={{ color: '#9ca3af' }}
            formatter={(value) => [`$${value.toFixed(2)}`, 'Precio']}
          />
          <Area
            type="monotone"
            dataKey="price"
            stroke="#06b6d4"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#colorPrice)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
