import { formatDateDisplay } from '../../utils/helpers';
import { getRiskColor, getRiskLevel } from '../../utils/riskUtils';

export default function ForecastCalendar({ forecast }) {
  if (!forecast || !forecast.forecast) {
    return (
      <div style={{ textAlign: 'center', padding: '32px', color: '#6b7280', fontSize: '14px' }}>
        No forecast data available
      </div>
    );
  }

  return (
    <div>
      <h3 style={{ fontSize: '17px', fontWeight: 700, color: '#d1d5db', marginBottom: '20px' }}>
        7-Day Risk Forecast
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px' }}>
        {forecast.forecast.map((day, index) => {
          const riskLevel = getRiskLevel(day.risk_score);
          const color = getRiskColor(riskLevel);

          return (
            <div
              key={index}
              style={{
                borderRadius: '14px',
                padding: '16px',
                background: 'rgba(255,255,255,0.05)',
                border: `2px solid ${color}55`,
                transition: 'box-shadow 0.2s',
              }}
              onMouseEnter={e => e.currentTarget.style.boxShadow = `0 4px 20px ${color}33`}
              onMouseLeave={e => e.currentTarget.style.boxShadow = 'none'}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                <div>
                  <p style={{ fontSize: '12px', color: '#6b7280', marginBottom: '3px' }}>Day {day.day}</p>
                  <p style={{ fontWeight: 600, color: '#d1d5db', fontSize: '14px' }}>{formatDateDisplay(day.date)}</p>
                </div>
                <div style={{ padding: '4px 10px', borderRadius: '20px', fontSize: '11px', fontWeight: 700, color: '#fff', background: color }}>
                  {riskLevel}
                </div>
              </div>

              {/* Risk Score Bar */}
              <div style={{ marginTop: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>
                  <span>Risk Score</span>
                  <span style={{ color: color, fontWeight: 700 }}>{Math.round(day.risk_score * 100)}%</span>
                </div>
                <div style={{ width: '100%', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', height: '6px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', borderRadius: '4px', backgroundColor: color, width: `${day.risk_score * 100}%`, transition: 'width 0.4s ease' }} />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}