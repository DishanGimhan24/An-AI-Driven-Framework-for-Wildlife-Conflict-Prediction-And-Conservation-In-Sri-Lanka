import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import Button from '../components/common/Button';
import Card from '../components/common/Card';
import DatePicker from '../components/common/DatePicker';
import ErrorMessage from '../components/common/ErrorMessage';
import Loading from '../components/common/Loading';
import DistrictSelector from '../components/features/DistrictSelector';
import Footer from '../components/layout/Footer';
import Navbar from '../components/layout/Navbar';
import { useHistorical } from '../hooks/useHistorical';
import { formatDateDisplay, getTodayDate } from '../utils/helpers';

const CHART_TOOLTIP_STYLE = {
  contentStyle: { background: '#111827', border: '1px solid rgba(255,255,255,0.18)', borderRadius: '10px', color: '#e5e7eb' },
};

export default function Historical() {
  const [district, setDistrict] = useState('');
  const [startDate, setStartDate] = useState('2009-01-01');
  const [endDate, setEndDate] = useState(getTodayDate());

  const { conflicts, statistics, loading, error, getConflicts, getStats } = useHistorical();

  useEffect(() => { loadData(); }, []);

  const loadData = async () => {
    await getConflicts(startDate, endDate, district || null);
    await getStats(startDate, endDate, district || null);
  };

  const getMonthlyTrends = () => {
    if (!statistics?.monthly_data?.length) return [];
    return statistics.monthly_data.map(item => ({ month: item.year_month, Conflicts: item.count }));
  };

  const getDistrictData = () => {
    // Prefer server-computed breakdown (always accurate, even when conflicts array is paginated/empty)
    if (statistics?.by_district && Object.keys(statistics.by_district).length > 0) {
      return Object.entries(statistics.by_district)
        .map(([d, count]) => ({ district: d, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 5);
    }
    // Fallback: compute from locally loaded conflicts
    if (!conflicts?.length) return [];
    const counts = {};
    conflicts.forEach(c => {
      const d = c.District || c.district || 'Unknown';
      counts[d] = (counts[d] || 0) + 1;
    });
    return Object.entries(counts).map(([d, count]) => ({ district: d, count })).sort((a, b) => b.count - a.count).slice(0, 5);
  };

  const getElephantDeaths = () => [
    { year: '2022', Deaths: 45 },
    { year: '2023', Deaths: 54 },
    { year: '2024', Deaths: 43 },
    { year: '2025', Deaths: 8 },
  ];

  const monthlyTrends = getMonthlyTrends();
  const districtData = getDistrictData();
  const elephantDeaths = getElephantDeaths();

  const statSummary = [
    { label: 'Total Conflicts', value: statistics?.total_conflicts || conflicts.length || 0, color: 'var(--emerald-400)', note: 'Selected period' },
    { label: 'Elephant Deaths (Train)', value: elephantDeaths.reduce((s, i) => s + i.Deaths, 0), color: '#f87171', note: '2022\u20132025' },
    { label: 'Districts Affected', value: statistics?.districts_affected ?? districtData.length, color: '#fbbf24', note: 'Unique districts' },
    { label: 'Avg Per Month', value: statistics?.avg_per_month || 0, color: '#818cf8', note: 'Conflict rate' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1 }}>
      <Navbar />

      <main style={{ flex: 1, maxWidth: '1400px', margin: '0 auto', padding: '32px 24px', width: '100%' }}>
        <div style={{ marginBottom: '32px' }}>
          <h1 className="page-title-gradient" style={{ fontSize: '2rem', marginBottom: '6px' }}>Historical Data &amp; Analytics</h1>
          <p style={{ color: '#9ca3af', fontSize: '15px' }}>Analyze past conflict patterns and trends</p>
        </div>

        {/* Filters */}
        <Card title="Filters" className="mb-6">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px', alignItems: 'end' }}>
            <DistrictSelector value={district} onChange={(e) => setDistrict(e.target.value)} label="District (Optional)" />
            <DatePicker label="Start Date" value={startDate} onChange={(e) => setStartDate(e.target.value)} max={endDate} />
            <DatePicker label="End Date" value={endDate} onChange={(e) => setEndDate(e.target.value)} min={startDate} max={getTodayDate()} />
            <Button onClick={loadData} variant="primary" style={{ width: '100%' }} loading={loading}>Apply Filters</Button>
          </div>
        </Card>

        {error && <ErrorMessage message={error} />}

        {loading ? (
          <Loading text="Loading historical data..." />
        ) : (
          <>
            {/* Summary Stats */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '28px' }}>
              {statSummary.map((s, i) => (
                <div key={i} className="glass-card" style={{ padding: '20px', borderTop: `3px solid ${s.color}` }}>
                  <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: '6px' }}>{s.label}</p>
                  <p style={{ fontSize: '2.2rem', fontWeight: 800, color: s.color, lineHeight: 1 }}>{s.value}</p>
                  <p style={{ fontSize: '11px', color: '#6b7280', marginTop: '6px' }}>{s.note}</p>
                </div>
              ))}
            </div>

            {/* Charts Row 1 */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px', marginBottom: '24px' }}>
              <Card title="Monthly Conflict Trends">
                {monthlyTrends.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={monthlyTrends}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#9ca3af' }} angle={-45} textAnchor="end" height={80} />
                      <YAxis tick={{ fill: '#9ca3af' }} />
                      <Tooltip {...CHART_TOOLTIP_STYLE} />
                      <Legend wrapperStyle={{ color: '#9ca3af' }} />
                      <Line type="monotone" dataKey="Conflicts" stroke="var(--emerald-500)" strokeWidth={2} dot={{ fill: 'var(--emerald-500)', r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '240px', color: '#6b7280', fontSize: '14px' }}>No data available</div>
                )}
              </Card>

              <Card title="Top 5 Districts by Conflicts">
                {districtData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={districtData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="district" tick={{ fontSize: 11, fill: '#9ca3af' }} />
                      <YAxis tick={{ fill: '#9ca3af' }} />
                      <Tooltip {...CHART_TOOLTIP_STYLE} />
                      <Legend wrapperStyle={{ color: '#9ca3af' }} />
                      <Bar dataKey="count" fill="#3b82f6" name="Conflicts" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '240px', color: '#6b7280', fontSize: '14px' }}>No data available</div>
                )}
              </Card>
            </div>

            {/* Charts Row 2 */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px', marginBottom: '24px' }}>
              <Card title="Elephant Deaths by Train Accidents">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={elephantDeaths}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="year" tick={{ fill: '#9ca3af' }} />
                    <YAxis tick={{ fill: '#9ca3af' }} />
                    <Tooltip {...CHART_TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ color: '#9ca3af' }} />
                    <Bar dataKey="Deaths" fill="#ef4444" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card title="Conflict Distribution by District">
                {districtData.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {districtData.map((item, index) => (
                      <div key={index} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderRadius: '10px', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}>
                        <span style={{ fontWeight: 600, color: '#d1d5db', fontSize: '14px' }}>{item.district}</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                          <div style={{ width: '100px', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', borderRadius: '4px', background: 'var(--emerald-500)', width: `${(item.count / districtData[0].count) * 100}%` }} />
                          </div>
                          <span style={{ fontWeight: 700, color: 'var(--emerald-400)', fontSize: '14px', minWidth: '32px', textAlign: 'right' }}>{item.count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '240px', color: '#6b7280', fontSize: '14px' }}>No data available</div>
                )}
              </Card>
            </div>

            {/* Conflicts Table */}
            <Card title={`Recent Conflicts (${conflicts.length} total)`}>
              {conflicts.length > 0 ? (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                    <thead>
                      <tr>
                        {['Date', 'Location', 'Coordinates'].map(h => (
                          <th key={h} style={{ padding: '10px 16px', textAlign: 'left', fontSize: '11px', fontWeight: 700, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.5px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {conflicts.slice(0, 10).map((conflict, index) => (
                        <tr key={index} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                          <td style={{ padding: '12px 16px', color: '#d1d5db', whiteSpace: 'nowrap' }}>
                            {conflict.Date ? formatDateDisplay(conflict.Date) : 'N/A'}
                          </td>
                          <td style={{ padding: '12px 16px', color: '#d1d5db', whiteSpace: 'nowrap' }}>
                            {conflict.District || conflict.district || 'Unknown'}
                          </td>
                          <td style={{ padding: '12px 16px', color: '#9ca3af', whiteSpace: 'nowrap', fontFamily: 'monospace', fontSize: '13px' }}>
                            {conflict.Latitude && conflict.Longitude
                              ? `${parseFloat(conflict.Latitude).toFixed(4)}, ${parseFloat(conflict.Longitude).toFixed(4)}`
                              : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {conflicts.length > 10 && (
                    <p style={{ textAlign: 'center', fontSize: '13px', color: '#6b7280', padding: '14px 0', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                      Showing 10 of {conflicts.length} conflicts
                    </p>
                  )}
                </div>
              ) : (
                <p style={{ textAlign: 'center', color: '#6b7280', padding: '32px 0', fontSize: '14px' }}>No conflicts found for the selected period</p>
              )}
            </Card>
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}

