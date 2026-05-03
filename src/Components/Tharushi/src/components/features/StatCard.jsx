import { TrendingUp, TrendingDown } from 'lucide-react';

export default function StatCard({ title, value, icon, color = 'primary', trend = null }) {
  const accentColor = {
    primary: 'var(--emerald-500)',
    danger: '#ef4444',
    warning: '#f59e0b',
    info: '#3b82f6',
  }[color] || 'var(--emerald-500)';

  const bgAccent = {
    primary: 'rgba(16,185,129,0.12)',
    danger: 'rgba(239,68,68,0.12)',
    warning: 'rgba(245,158,11,0.12)',
    info: 'rgba(59,130,246,0.12)',
  }[color] || 'rgba(16,185,129,0.12)';

  return (
    <div
      className="glass-card"
      style={{ padding: '20px', borderTop: `3px solid ${accentColor}` }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ flex: 1 }}>
          <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: '6px', fontWeight: 500 }}>{title}</p>
          <p style={{ fontSize: '2.2rem', fontWeight: 800, color: 'white', lineHeight: 1 }}>{value}</p>

          {trend && (
            <div style={{ display: 'flex', alignItems: 'center', marginTop: '8px', fontSize: '13px', gap: '4px' }}>
              {trend.direction === 'up' ? (
                <TrendingUp size={16} style={{ color: '#f87171' }} />
              ) : (
                <TrendingDown size={16} style={{ color: 'var(--emerald-400)' }} />
              )}
              <span style={{ color: trend.direction === 'up' ? '#f87171' : 'var(--emerald-400)', fontWeight: 600 }}>{trend.value}</span>
              <span style={{ color: '#9ca3af' }}>vs last month</span>
            </div>
          )}
        </div>

        {icon && (
          <div style={{
            padding: '14px',
            borderRadius: '14px',
            background: bgAccent,
            color: accentColor,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}