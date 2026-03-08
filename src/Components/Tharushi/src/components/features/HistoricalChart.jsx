import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const TOOLTIP_STYLE = {
  contentStyle: { background: '#111827', border: '1px solid rgba(255,255,255,0.18)', borderRadius: '10px', color: '#e5e7eb' },
};

export default function HistoricalChart({ data = [], type = 'line', title }) {
  if (!data || data.length === 0) {
    return (
      <div className="glass-card" style={{ padding: '24px' }}>
        {title && <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#d1d5db', marginBottom: '16px' }}>{title}</h3>}
        <div style={{ textAlign: 'center', padding: '32px 0', color: '#6b7280', fontSize: '14px' }}>No data available</div>
      </div>
    );
  }

  return (
    <div className="glass-card" style={{ padding: '24px' }}>
      {title && <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#d1d5db', marginBottom: '16px' }}>{title}</h3>}

      <ResponsiveContainer width="100%" height={300}>
        {type === 'line' ? (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis dataKey="name" tick={{ fill: '#9ca3af' }} />
            <YAxis tick={{ fill: '#9ca3af' }} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ color: '#9ca3af' }} />
            <Line type="monotone" dataKey="value" stroke="var(--emerald-500)" strokeWidth={2} dot={{ fill: 'var(--emerald-500)', r: 3 }} />
          </LineChart>
        ) : (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis dataKey="name" tick={{ fill: '#9ca3af' }} />
            <YAxis tick={{ fill: '#9ca3af' }} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ color: '#9ca3af' }} />
            <Bar dataKey="value" fill="var(--emerald-600)" radius={[6, 6, 0, 0]} />
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}