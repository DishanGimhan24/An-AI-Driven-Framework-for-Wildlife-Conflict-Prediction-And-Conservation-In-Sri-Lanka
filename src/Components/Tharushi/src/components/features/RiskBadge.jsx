export default function RiskBadge({ level, size = 'medium' }) {
  const colorMap = {
    HIGH: { border: '#ef4444', color: '#f87171', bg: 'rgba(239,68,68,0.12)' },
    MEDIUM: { border: '#f59e0b', color: '#fbbf24', bg: 'rgba(245,158,11,0.12)' },
    LOW: { border: 'var(--emerald-500)', color: 'var(--emerald-400)', bg: 'rgba(16,185,129,0.12)' },
  };

  const sizeMap = {
    small: { padding: '4px 10px', fontSize: '12px' },
    medium: { padding: '6px 14px', fontSize: '13px' },
    large: { padding: '10px 22px', fontSize: '16px' },
  };

  const { border, color, bg } = colorMap[level] || { border: '#6b7280', color: '#9ca3af', bg: 'rgba(107,114,128,0.12)' };
  const sizeStyle = sizeMap[size] || sizeMap.medium;

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      fontWeight: 700,
      borderRadius: '20px',
      border: `2px solid ${border}`,
      color,
      background: bg,
      letterSpacing: '0.5px',
      ...sizeStyle,
    }}>
      {level}
    </span>
  );
}