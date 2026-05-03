import { useCallback, useEffect, useState } from 'react';
import {
  CartesianGrid, Legend, Line, LineChart,
  PolarAngleAxis, PolarGrid, PolarRadiusAxis,
  Radar, RadarChart, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from 'recharts';
import { getSeasonalProfile, getStressCorrelation, getStressSummary } from '../api/environmentalStressAPI';
import Card from '../components/common/Card';
import ErrorMessage from '../components/common/ErrorMessage';
import Loading from '../components/common/Loading';
import DistrictSelector from '../components/features/DistrictSelector';
import Footer from '../components/layout/Footer';
import Navbar from '../components/layout/Navbar';

const CHART_TOOLTIP_STYLE = {
  contentStyle: { background: '#111827', border: '1px solid rgba(255,255,255,0.18)', borderRadius: '10px', color: '#e5e7eb' },
};

const STRESS_COLOR = { HIGH: '#ef4444', MEDIUM: '#f59e0b', LOW: '#10b981' };

const fmtSign = (val, decimals = 2, unit = '') => {
  if (val == null) return 'N/A';
  const sign = val >= 0 ? '+' : '';
  return `${sign}${val.toFixed(decimals)}${unit}`;
};

const ndviTileColor = (val) => {
  if (val < -0.10) return 'rgba(239,68,68,0.75)';
  if (val < -0.05) return 'rgba(245,158,11,0.75)';
  if (val <= 0.05) return 'rgba(59,130,246,0.5)';
  return 'rgba(16,185,129,0.75)';
};

export default function EnvironmentalStress() {
  const [summaryData, setSummaryData] = useState(null);
  const [profileData, setProfileData] = useState(null);
  const [correlationData, setCorrelationData] = useState(null);
  const [selectedDistrict, setSelectedDistrict] = useState('Kandy');
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [profileLoading, setProfileLoading] = useState(false);
  const [correlationLoading, setCorrelationLoading] = useState(false);
  const [error, setError] = useState('');

  const loadSummary = useCallback(async () => {
    setSummaryLoading(true);
    try {
      const data = await getStressSummary();
      setSummaryData(data);
    } catch (err) {
      setError(typeof err === 'string' ? err : 'Failed to load stress summary');
    } finally {
      setSummaryLoading(false);
    }
  }, []);

  const loadProfile = useCallback(async (district) => {
    setProfileLoading(true);
    try {
      const data = await getSeasonalProfile(district);
      setProfileData(data);
    } catch (err) {
      setError(typeof err === 'string' ? err : 'Failed to load seasonal profile');
    } finally {
      setProfileLoading(false);
    }
  }, []);

  const loadCorrelation = useCallback(async () => {
    setCorrelationLoading(true);
    try {
      const data = await getStressCorrelation();
      setCorrelationData(data);
    } catch (err) {
      setError(typeof err === 'string' ? err : 'Failed to load correlation data');
    } finally {
      setCorrelationLoading(false);
    }
  }, []);

  useEffect(() => { loadSummary(); }, [loadSummary]);
  useEffect(() => { loadCorrelation(); }, [loadCorrelation]);
  useEffect(() => { loadProfile(selectedDistrict); }, [selectedDistrict, loadProfile]);

  const avgNdvi = summaryData?.districts?.length
    ? summaryData.districts.reduce((sum, d) => sum + d.ndvi_anomaly, 0) / summaryData.districts.length
    : null;

  const sortedDistricts = summaryData?.districts
    ? [...summaryData.districts].sort((a, b) => b.stress_score - a.stress_score)
    : [];

  const radarData = profileData?.profile
    ? profileData.profile.map(m => ({
        month: m.month_name,
        'NDVI (×100)': Math.round(m.avg_ndvi * 100),
        'Sightings': m.gbif_sighting_count,
        'Rainfall (÷5)': Math.round(m.avg_rainfall_30day / 5),
      }))
    : [];

  const pearsonR = correlationData?.pearson_r;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1 }}>
      <Navbar />

      <main style={{ flex: 1, maxWidth: '1400px', margin: '0 auto', padding: '32px 24px', width: '100%' }}>
        <div style={{ marginBottom: '32px' }}>
          <h1 className="page-title-gradient" style={{ fontSize: '2rem', marginBottom: '6px' }}>Environmental Stress Index</h1>
          <p style={{ color: '#9ca3af', fontSize: '15px' }}>Ecological pressure driving elephant movement</p>
        </div>

        {error && <ErrorMessage message={error} />}

        {/* Section 1: KPI Cards */}
        {summaryLoading ? (
          <Loading text="Loading stress summary..." />
        ) : summaryData && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '28px' }}>
            <div className="glass-card" style={{ padding: '20px', borderTop: '3px solid var(--emerald-400)' }}>
              <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: '6px' }}>National Avg Stress Score</p>
              <p style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--emerald-400)', lineHeight: 1 }}>
                {summaryData.national_avg_stress?.toFixed(1)}
              </p>
              <p style={{ fontSize: '11px', color: '#6b7280', marginTop: '6px' }}>0–100 scale</p>
            </div>

            <div className="glass-card" style={{ padding: '20px', borderTop: '3px solid #ef4444' }}>
              <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: '6px' }}>HIGH-stress Districts</p>
              <p style={{ fontSize: '2.2rem', fontWeight: 800, color: '#ef4444', lineHeight: 1 }}>
                {summaryData.high_stress_count}
              </p>
              <p style={{ fontSize: '11px', color: '#6b7280', marginTop: '6px' }}>Of 25 districts</p>
            </div>

            <div className="glass-card" style={{ padding: '20px', borderTop: '3px solid #f59e0b' }}>
              <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: '6px' }}>Most Stressed</p>
              <p style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f59e0b', lineHeight: 1.2 }}>
                {summaryData.most_stressed_district}
              </p>
              <p style={{ fontSize: '11px', color: '#6b7280', marginTop: '6px' }}>Highest stress score</p>
            </div>

            <div className="glass-card" style={{ padding: '20px', borderTop: '3px solid #818cf8' }}>
              <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: '6px' }}>National NDVI Anomaly</p>
              <p style={{ fontSize: '2.2rem', fontWeight: 800, color: avgNdvi >= 0 ? '#10b981' : '#ef4444', lineHeight: 1 }}>
                {fmtSign(avgNdvi)}
              </p>
              <p style={{ fontSize: '11px', color: '#6b7280', marginTop: '6px' }}>Avg across districts</p>
            </div>
          </div>
        )}

        {/* Section 2: District Stress Rankings */}
        <div style={{ marginBottom: '24px' }}>
          <Card title="District Stress Rankings">
            {summaryLoading ? (
              <Loading text="Loading district rankings..." />
            ) : sortedDistricts.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {sortedDistricts.map((item, i) => (
                  <div
                    key={i}
                    style={{
                      display: 'flex', alignItems: 'center',
                      padding: '10px 14px', borderRadius: '10px',
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.07)',
                      gap: '12px',
                    }}
                  >
                    <span style={{ fontWeight: 600, color: '#d1d5db', fontSize: '14px', minWidth: '120px' }}>
                      {item.district}
                    </span>
                    <div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', borderRadius: '4px',
                        background: STRESS_COLOR[item.stress_label] || '#9ca3af',
                        width: `${Math.min(item.stress_score, 100)}%`,
                      }} />
                    </div>
                    <span style={{ fontWeight: 700, color: STRESS_COLOR[item.stress_label] || '#9ca3af', fontSize: '14px', minWidth: '40px', textAlign: 'right' }}>
                      {item.stress_score?.toFixed(1)}
                    </span>
                    <span style={{ fontSize: '13px', color: item.ndvi_anomaly >= 0 ? '#10b981' : '#ef4444', minWidth: '60px', textAlign: 'right' }}>
                      {fmtSign(item.ndvi_anomaly, 2)}
                    </span>
                    <span style={{ fontSize: '13px', color: '#9ca3af', minWidth: '80px', textAlign: 'right' }}>
                      {fmtSign(item.rainfall_anomaly, 1, ' mm')}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '120px', color: '#6b7280', fontSize: '14px' }}>
                No data available
              </div>
            )}
          </Card>
        </div>

        {/* Section 3: NDVI Tile Grid + Seasonal Radar */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
          {/* Left: NDVI Anomaly Tile Grid */}
          <Card title="NDVI Anomaly by District">
            {summaryLoading ? (
              <Loading text="Loading NDVI data..." />
            ) : summaryData?.districts ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '6px' }}>
                  {summaryData.districts.map((d, i) => (
                    <div
                      key={i}
                      style={{
                        background: ndviTileColor(d.ndvi_anomaly),
                        borderRadius: '8px',
                        padding: '8px 4px',
                        textAlign: 'center',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '2px',
                      }}
                    >
                      <span style={{ fontSize: '10px', fontWeight: 700, color: '#fff', lineHeight: 1.2 }}>{d.district}</span>
                      <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.9)' }}>{fmtSign(d.ndvi_anomaly, 2)}</span>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: '12px', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                  {[
                    { color: 'rgba(239,68,68,0.75)', label: 'Severely stressed (< -0.10)' },
                    { color: 'rgba(245,158,11,0.75)', label: 'Moderately stressed' },
                    { color: 'rgba(59,130,246,0.5)', label: 'Near baseline' },
                    { color: 'rgba(16,185,129,0.75)', label: 'Above baseline (> +0.05)' },
                  ].map((leg, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                      <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: leg.color, flexShrink: 0 }} />
                      <span style={{ fontSize: '11px', color: '#9ca3af' }}>{leg.label}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '240px', color: '#6b7280', fontSize: '14px' }}>
                No data available
              </div>
            )}
          </Card>

          {/* Right: Seasonal Radar Chart */}
          <Card title="Seasonal Stress Profile">
            <DistrictSelector
              value={selectedDistrict}
              onChange={(e) => setSelectedDistrict(e.target.value)}
              label="District"
            />
            {profileLoading ? (
              <Loading text="Loading seasonal profile..." />
            ) : radarData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.15)" />
                    <PolarAngleAxis dataKey="month" tick={{ fontSize: 11, fill: '#9ca3af' }} />
                    <PolarRadiusAxis tick={{ fontSize: 10, fill: '#9ca3af' }} />
                    <Tooltip {...CHART_TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ color: '#9ca3af' }} />
                    <Radar name="NDVI (×100)" dataKey="NDVI (×100)" stroke="#10b981" fill="#10b981" fillOpacity={0.15} />
                    <Radar name="Sightings" dataKey="Sightings" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} />
                    <Radar name="Rainfall (÷5)" dataKey="Rainfall (÷5)" stroke="#60a5fa" fill="#60a5fa" fillOpacity={0.15} />
                  </RadarChart>
                </ResponsiveContainer>
                <p style={{ fontSize: '12px', color: '#6b7280', textAlign: 'center', marginTop: '8px', fontStyle: 'italic' }}>
                  NDVI ×100 and Rainfall ÷5 for axis alignment
                </p>
              </>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '240px', color: '#6b7280', fontSize: '14px' }}>
                No data available
              </div>
            )}
          </Card>
        </div>

        {/* Section 4: Stress–Conflict Correlation Chart */}
        <Card title="Stress–Conflict Correlation">
          {correlationLoading ? (
            <Loading text="Loading correlation data..." />
          ) : correlationData?.series?.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={correlationData.series}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis
                    dataKey="year_month"
                    tick={{ fontSize: 11, fill: '#9ca3af' }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    interval={2}
                  />
                  <YAxis
                    yAxisId="left"
                    tick={{ fill: '#9ca3af' }}
                    label={{ value: 'NDVI Anomaly', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 12 }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    tick={{ fill: '#9ca3af' }}
                    label={{ value: 'Elephant Sightings', angle: 90, position: 'insideRight', fill: '#9ca3af', fontSize: 12 }}
                  />
                  <Tooltip {...CHART_TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ color: '#9ca3af' }} />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="ndvi_anomaly"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={false}
                    name="NDVI Anomaly"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="gbif_count"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name="Elephant Sightings"
                  />
                </LineChart>
              </ResponsiveContainer>

              {pearsonR != null && (
                <div style={{ marginTop: '16px', padding: '14px', background: 'rgba(255,255,255,0.04)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)' }}>
                  <p style={{ fontSize: '14px', color: '#d1d5db', margin: 0 }}>
                    Pearson r (NDVI anomaly vs. sightings):{' '}
                    <span style={{ fontWeight: 700, color: pearsonR < -0.2 ? '#ef4444' : pearsonR > 0.2 ? '#10b981' : '#9ca3af' }}>
                      {pearsonR.toFixed(2)}
                    </span>
                  </p>
                  {pearsonR < -0.2 && (
                    <p style={{ fontSize: '13px', color: '#9ca3af', marginTop: '6px', marginBottom: 0, fontStyle: 'italic' }}>
                      Negative correlation confirms that vegetation stress is associated with increased elephant sightings near settlements.
                    </p>
                  )}
                </div>
              )}
            </>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '240px', color: '#6b7280', fontSize: '14px' }}>
              No data available
            </div>
          )}
        </Card>
      </main>

      <Footer />
    </div>
  );
}
