import { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { getMonthName } from '../../utils/dateUtils';
import { getRiskColor } from '../../utils/riskUtils';

export default function MonthlyCalendar({ predictions = [] }) {
  const [currentDate, setCurrentDate] = useState(new Date());

  const year = currentDate.getFullYear();
  const month = currentDate.getMonth();
  const monthName = getMonthName(currentDate);

  // Get first day of month and number of days
  const firstDay = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  // Create array of day objects
  const days = [];
  for (let i = 0; i < firstDay; i++) {
    days.push(null);
  }
  for (let day = 1; day <= daysInMonth; day++) {
    days.push(day);
  }

  // Get risk for specific date
  const getRiskForDate = (day) => {
    const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
    return predictions.find(p => p.date === dateStr);
  };

  // Navigate months
  const previousMonth = () => {
    setCurrentDate(new Date(year, month - 1, 1));
  };

  const nextMonth = () => {
    setCurrentDate(new Date(year, month + 1, 1));
  };

  const weekDays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  return (
    <div style={{ padding: '24px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
        <button
          onClick={previousMonth}
          style={{ padding: '8px', borderRadius: '50%', background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)', cursor: 'pointer', color: '#9ca3af', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.15)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.08)'}
        >
          <ChevronLeft size={20} />
        </button>

        <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#d1d5db' }}>{monthName} {year}</h3>

        <button
          onClick={nextMonth}
          style={{ padding: '8px', borderRadius: '50%', background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)', cursor: 'pointer', color: '#9ca3af', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.15)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.08)'}
        >
          <ChevronRight size={20} />
        </button>
      </div>

      {/* Week days */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '8px', marginBottom: '8px' }}>
        {weekDays.map(day => (
          <div key={day} style={{ padding: '8px 0', fontSize: '12px', fontWeight: 600, textAlign: 'center', color: '#6b7280' }}>
            {day}
          </div>
        ))}
      </div>

      {/* Calendar days */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '8px' }}>
        {days.map((day, index) => {
          if (!day) return <div key={index} style={{ aspectRatio: '1' }} />;

          const risk = getRiskForDate(day);
          const color = risk ? getRiskColor(risk.risk_level) : 'rgba(255,255,255,0.08)';
          const hasRisk = !!risk;

          return (
            <div
              key={index}
              style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                padding: '8px', aspectRatio: '1', borderRadius: '8px', cursor: 'pointer',
                border: `2px solid ${hasRisk ? color : 'rgba(255,255,255,0.08)'}`,
                background: hasRisk ? `${color}22` : 'rgba(255,255,255,0.03)',
                transition: 'all 0.2s',
              }}
              title={hasRisk ? `${risk.risk_level} - ${Math.round(risk.risk_score * 100)}%` : 'No data'}
            >
              <span style={{ fontSize: '13px', fontWeight: 600, color: hasRisk ? '#d1d5db' : '#6b7280' }}>{day}</span>
              {hasRisk && (
                <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: color, marginTop: '4px' }} />
              )}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '24px', fontSize: '13px' }}>
          {['HIGH', 'MEDIUM', 'LOW'].map(level => (
            <div key={level} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: getRiskColor(level) }} />
              <span style={{ color: '#9ca3af' }}>{level.charAt(0) + level.slice(1).toLowerCase()} Risk</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}