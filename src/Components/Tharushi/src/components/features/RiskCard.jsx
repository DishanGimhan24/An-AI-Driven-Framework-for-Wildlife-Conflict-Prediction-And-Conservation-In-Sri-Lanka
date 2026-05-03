import { getRiskPercentage } from '../../utils/helpers';
import { getRiskBgColor, getRiskColor, getRiskDescription, getRiskText } from '../../utils/riskUtils';

export default function RiskCard({ riskScore, riskLevel, showDescription = true, size = 'md' }) {
  const percentage = getRiskPercentage(riskScore);
  const color = getRiskColor(riskLevel);
  const text = getRiskText(riskLevel);
  const description = getRiskDescription(riskLevel);

  // Map light colors to dark-theme equivalents
  const darkColor = {
    HIGH: '#ef4444',
    MEDIUM: '#f59e0b',
    LOW: 'var(--emerald-500)',
  }[riskLevel] || '#6b7280';

  const padMap = { sm: '16px', md: '24px', lg: '32px' };

  return (
    <div
      className="glass-card"
      style={{
        padding: padMap[size] || '24px',
        borderLeft: `4px solid ${darkColor}`,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#d1d5db', margin: 0 }}>Risk Level</h3>
        <div style={{
          padding: '6px 14px',
          borderRadius: '20px',
          background: `${darkColor}22`,
          border: `2px solid ${darkColor}`,
          color: darkColor,
          fontWeight: 700,
          fontSize: '14px',
        }}>
          {text}
        </div>
      </div>

      {/* Risk Score Circle */}
      <div style={{ display: 'flex', justifyContent: 'center', margin: '24px 0' }}>
        <div style={{ position: 'relative' }}>
          <svg style={{ transform: 'rotate(-90deg)', width: '128px', height: '128px' }}>
            <circle cx="64" cy="64" r="56" stroke="rgba(255,255,255,0.1)" strokeWidth="8" fill="none" />
            <circle
              cx="64" cy="64" r="56"
              stroke={darkColor}
              strokeWidth="8"
              fill="none"
              strokeDasharray={`${percentage * 3.52} 352`}
              strokeLinecap="round"
            />
          </svg>
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: darkColor }}>{percentage}%</div>
              <div style={{ fontSize: '11px', color: '#9ca3af' }}>Risk Score</div>
            </div>
          </div>
        </div>
      </div>

      {showDescription && (
        <p style={{ fontSize: '13px', color: '#9ca3af', textAlign: 'center' }}>{description}</p>
      )}
    </div>
  );
}