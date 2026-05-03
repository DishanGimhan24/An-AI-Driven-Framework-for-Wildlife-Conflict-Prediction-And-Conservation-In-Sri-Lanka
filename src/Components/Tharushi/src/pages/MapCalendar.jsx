import { useEffect, useState } from 'react';
import { Map, Calendar } from 'lucide-react';
import Button from '../components/common/Button';
import Card from '../components/common/Card';
import DatePicker from '../components/common/DatePicker';
import ErrorMessage from '../components/common/ErrorMessage';
import Loading from '../components/common/Loading';
import CitySelector from '../components/features/CitySelector';
import DistrictSelector from '../components/features/DistrictSelector';
import MonthlyCalendar from '../components/features/MonthlyCalendar';
import RiskHeatmap from '../components/features/RiskHeatmap';
import Footer from '../components/layout/Footer';
import Navbar from '../components/layout/Navbar';
import { getHistoricalConflicts } from '../api/historicalAPI';
import { useCityHeatmap } from '../hooks/useCityHeatmap';
import { useForecast } from '../hooks/useForecast';
import { useHeatmap } from '../hooks/useHeatmap';
import { DISTRICT_COORDINATES } from '../utils/constants';
import { getTodayDate } from '../utils/helpers';

const TAB_STYLE_ACTIVE = {
  padding: '10px 22px', borderRadius: '30px', fontWeight: 600, fontSize: '14px', cursor: 'pointer', border: 'none',
  background: 'linear-gradient(135deg, var(--emerald-500), var(--emerald-600))', color: '#fff',
  boxShadow: '0 4px 14px rgba(16,185,129,0.3)',
};
const TAB_STYLE_INACTIVE = {
  padding: '10px 22px', borderRadius: '30px', fontWeight: 500, fontSize: '14px', cursor: 'pointer',
  background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', color: '#9ca3af',
};
const VIEW_BTN_ACTIVE = {
  padding: '8px 18px', borderRadius: '20px', fontWeight: 600, fontSize: '13px', cursor: 'pointer', border: 'none',
  background: 'linear-gradient(135deg, var(--emerald-500), var(--emerald-600))', color: '#fff',
};
const VIEW_BTN_INACTIVE = {
  padding: '8px 18px', borderRadius: '20px', fontWeight: 500, fontSize: '13px', cursor: 'pointer',
  background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', color: '#9ca3af',
};

// Assign each raw incident (lat/lon) to the nearest district by straight-line
// distance, then derive a risk level from the per-district incident count.
// This produces genuinely date-specific results for past dates instead of the
// ML model's season-averaged predictions.
function buildHistoricalDistrictData(conflicts, date) {
  const districtNames = Object.keys(DISTRICT_COORDINATES);
  const counts = Object.fromEntries(districtNames.map(d => [d, 0]));

  conflicts.forEach(c => {
    const lat = parseFloat(c.latitude);
    const lon = parseFloat(c.longitude);
    if (isNaN(lat) || isNaN(lon)) return;
    let nearest = null;
    let minDist = Infinity;
    districtNames.forEach(name => {
      const co = DISTRICT_COORDINATES[name];
      const dist = Math.hypot(lat - co.lat, lon - co.lng);
      if (dist < minDist) { minDist = dist; nearest = name; }
    });
    if (nearest) counts[nearest]++;
  });

  const districts = districtNames.map(name => {
    const n = counts[name];
    return {
      district: name,
      risk_level: n >= 3 ? 'HIGH' : n >= 1 ? 'MEDIUM' : 'LOW',
      risk_score: n >= 3 ? 0.85 : n >= 1 ? 0.55 : 0.15,
      incident_count: n,
    };
  });

  const high = districts.filter(d => d.risk_level === 'HIGH').length;
  const medium = districts.filter(d => d.risk_level === 'MEDIUM').length;
  const low = districts.filter(d => d.risk_level === 'LOW').length;

  return {
    date,
    type: 'historical',
    districts,
    summary: { high_risk_districts: high, medium_risk_districts: medium, low_risk_districts: low },
  };
}

export default function MapCalendar() {
  const [activeTab, setActiveTab] = useState('map');
  const [selectedDistrict, setSelectedDistrict] = useState('');
  const [selectedCity, setSelectedCity] = useState('');
  const [selectedDate, setSelectedDate] = useState(getTodayDate());
  const [calendarPredictions, setCalendarPredictions] = useState([]);
  const [viewMode, setViewMode] = useState('district');
  // Holds transformed historical incident data when a past date is selected;
  // null when showing ML predictions for today / future.
  const [overrideDistrictData, setOverrideDistrictData] = useState(null);
  const [loadingHistorical, setLoadingHistorical] = useState(false);

  const { getForecast, loading: forecasting } = useForecast();
  const { districtData, loading: loadingHeatmap, error: heatmapError, loadDistrictHeatmap } = useHeatmap();
  const { cityHeatmapData, loading: loadingCityHeatmap, loadCityHeatmap } = useCityHeatmap();

  // Route to the correct data source based on whether the date is past, today, or future.
  //
  // Past  → /api/conflicts for the FULL MONTH containing the selected date.
  //         The GBIF occurrence dataset has ~1,300 records (2007–2026); a single-day
  //         query almost always returns 0 because data is sparse and the backend's
  //         end_date filter uses midnight-comparison (records with timestamps like
  //         "2025-06-15T10:30" are excluded by <= 2025-06-15 00:00:00).  Using the
  //         full month (start = 1st, end = 1st of NEXT month) captures all records
  //         and gives genuine per-month variation (Aug 2024 = 68 incidents, May 2024
  //         = 2 incidents, etc.).  Falls back to ML prediction when 0 incidents found.
  //
  // Today → /api/heatmap/districts  (current ML prediction)
  // Future → /api/heatmap/districts  (seasonal ML forecast)
  const loadMapDataForDate = async (date) => {
    const today = getTodayDate();

    if (date < today) {
      setLoadingHistorical(true);
      setOverrideDistrictData(null);
      try {
        // Build full-month window.  Use first day of NEXT month as end_date so
        // the backend's '<= midnight' comparison includes all records in the month,
        // including those on the last day with time components (e.g. "2024-08-31T15:30").
        const [y, m] = date.split('-').map(Number);
        const monthStr = String(m).padStart(2, '0');
        const firstDay = `${y}-${monthStr}-01`;
        const nextM = m === 12 ? 1 : m + 1;
        const nextY = m === 12 ? y + 1 : y;
        const firstDayNextMonth = `${nextY}-${String(nextM).padStart(2, '0')}-01`;

        console.log(`Fetching risk data for date: ${date} (past → /api/conflicts, window: ${firstDay} to ${firstDayNextMonth})`);
        const resp = await getHistoricalConflicts(firstDay, firstDayNextMonth);
        const conflicts = resp.data?.conflicts ?? [];
        console.log(`Fetching risk data for date: ${date}, raw conflict count: ${conflicts.length}, response:`, resp.data);

        if (conflicts.length === 0) {
          // No recorded incidents for this month — use ML seasonal estimate as fallback.
          console.warn(`No historical incidents for ${firstDay}–${firstDayNextMonth}; falling back to ML prediction`);
          setOverrideDistrictData(null);
          const result = await loadDistrictHeatmap(date);
          console.log(`Fetching risk data for date: ${date} (ML fallback), response:`, result);
        } else {
          const historical = buildHistoricalDistrictData(conflicts, date);
          console.log(`Fetching risk data for date: ${date}, transformed historical district data:`, historical);
          setOverrideDistrictData(historical);
        }
      } catch (err) {
        console.error(`Historical fetch failed for ${date}, falling back to ML prediction:`, err);
        setOverrideDistrictData(null);
        const result = await loadDistrictHeatmap(date);
        console.log(`Fetching risk data for date: ${date} (error fallback ML), response:`, result);
      } finally {
        setLoadingHistorical(false);
      }
    } else {
      setOverrideDistrictData(null);
      console.log(`Fetching risk data for date: ${date} (${date === today ? 'today' : 'future'} → /api/heatmap/districts)`);
      const result = await loadDistrictHeatmap(date);
      console.log(`Fetching risk data for date: ${date}, response:`, result);
    }
  };

  useEffect(() => {
    if (activeTab === 'map' && viewMode === 'district') {
      loadMapDataForDate(selectedDate);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, viewMode]);

  useEffect(() => {
    if (activeTab === 'map' && viewMode === 'city' && selectedDistrict) {
      loadCityHeatmap(selectedDistrict, selectedDate);
    }
  }, [selectedDistrict, selectedDate, viewMode, activeTab]);

  const handleGenerateCalendar = async () => {
    if (!selectedDistrict) { alert('Please select a district first'); return; }
    const coords = DISTRICT_COORDINATES[selectedDistrict];
    if (!coords || !coords.lat || !coords.lng) { alert(`Coordinates not found for ${selectedDistrict}`); return; }
    const result = await getForecast(coords.lat, coords.lng, 30, selectedDate);
    if (result && result.forecast) {
      setCalendarPredictions(result.forecast.map(day => ({ date: day.date, risk_score: day.risk_score, risk_level: day.risk_level })));
    }
  };

  const handleDateChange = async (e) => {
    const newDate = e.target.value;
    setSelectedDate(newDate);
    if (activeTab === 'map') {
      if (viewMode === 'district') await loadMapDataForDate(newDate);
      else if (viewMode === 'city' && selectedDistrict) await loadCityHeatmap(selectedDistrict, newDate);
    }
  };

  const currentHeatmapData = viewMode === 'city' ? cityHeatmapData : (overrideDistrictData || districtData);
  const currentLoading = viewMode === 'city' ? loadingCityHeatmap : (loadingHistorical || loadingHeatmap);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1 }}>
      <Navbar />

      <main style={{ flex: 1, maxWidth: '1400px', margin: '0 auto', padding: '32px 24px', width: '100%' }}>
        <div style={{ marginBottom: '32px' }}>
          <h1 className="page-title-gradient" style={{ fontSize: '2rem', marginBottom: '6px' }}>Risk Map &amp; Calendar</h1>
          <p style={{ color: '#9ca3af', fontSize: '15px' }}>Visualize elephant conflict risk zones across Sri Lanka</p>
        </div>

        {/* Tab buttons */}
        <div style={{ display: 'flex', gap: '12px', marginBottom: '28px' }}>
          <button style={activeTab === 'map' ? TAB_STYLE_ACTIVE : TAB_STYLE_INACTIVE} onClick={() => setActiveTab('map')}><span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Map size={16} /> Risk Map</span></button>
          <button style={activeTab === 'calendar' ? TAB_STYLE_ACTIVE : TAB_STYLE_INACTIVE} onClick={() => setActiveTab('calendar')}><span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Calendar size={16} /> Risk Calendar</span></button>
        </div>

        {/* Map Tab */}
        {activeTab === 'map' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* View Mode Toggle */}
            <Card>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
                <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#d1d5db' }}>View Mode</h3>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button style={viewMode === 'district' ? VIEW_BTN_ACTIVE : VIEW_BTN_INACTIVE} onClick={() => setViewMode('district')}>District View</button>
                  <button style={viewMode === 'city' ? VIEW_BTN_ACTIVE : VIEW_BTN_INACTIVE} onClick={() => setViewMode('city')}>City View</button>
                </div>
              </div>
            </Card>

            {/* Controls */}
            <Card title="Map Settings">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
                <DatePicker label="Select Date" value={selectedDate} onChange={handleDateChange} />
                {viewMode === 'city' && (
                  <>
                    <DistrictSelector value={selectedDistrict} onChange={(e) => setSelectedDistrict(e.target.value)} label="Select District" />
                    <CitySelector district={selectedDistrict} value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)} label="Select City/Division" />
                  </>
                )}
              </div>
            </Card>

            {/* Map */}
            <Card>
              {currentLoading ? (
                <Loading message="Loading map data..." />
              ) : heatmapError ? (
                <ErrorMessage message={heatmapError} />
              ) : currentHeatmapData ? (
                <>
                  <div style={{ marginBottom: '16px' }}>
                    <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#d1d5db' }}>
                      {viewMode === 'city' ? `${selectedDistrict} – City Level Risk` : 'District Level Risk Map'}
                    </h3>
                    <p style={{ fontSize: '13px', color: '#9ca3af', marginTop: '4px' }}>
                      Date: {selectedDate}
                      {overrideDistrictData && (
                        <span style={{ marginLeft: '8px', color: '#f59e0b', fontSize: '12px' }}>
                          (Historical Incidents)
                        </span>
                      )}
                    </p>
                  </div>
                  <RiskHeatmap districtData={currentHeatmapData} viewMode={viewMode} />
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: '48px 0', color: '#6b7280', fontSize: '14px' }}>
                  {viewMode === 'city' ? 'Select a district to view city-level risks' : 'Loading map...'}
                </div>
              )}
            </Card>

            {/* Risk Summary */}
            {currentHeatmapData && currentHeatmapData.summary && (
              <Card title="Risk Summary">
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                  {[
                    { label: 'High Risk', key: 'high_risk_cities', key2: 'high_risk_districts', color: '#ef4444', bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.25)' },
                    { label: 'Medium Risk', key: 'medium_risk_cities', key2: 'medium_risk_districts', color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.25)' },
                    { label: 'Low Risk', key: 'low_risk_cities', key2: 'low_risk_districts', color: 'var(--emerald-400)', bg: 'rgba(16,185,129,0.1)', border: 'rgba(16,185,129,0.25)' },
                  ].map(item => (
                    <div key={item.label} style={{ textAlign: 'center', padding: '20px', borderRadius: '12px', background: item.bg, border: `1px solid ${item.border}` }}>
                      <p style={{ fontSize: '2rem', fontWeight: 800, color: item.color }}>
                        {currentHeatmapData.summary[item.key] || currentHeatmapData.summary[item.key2] || 0}
                      </p>
                      <p style={{ fontSize: '13px', color: '#9ca3af', marginTop: '4px' }}>{item.label}</p>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        )}

        {/* Calendar Tab */}
        {activeTab === 'calendar' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <Card title="Calendar Settings">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                <DistrictSelector value={selectedDistrict} onChange={(e) => setSelectedDistrict(e.target.value)} label="Select District" />
                <DatePicker value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)} label="Start Date" min={getTodayDate()} />
              </div>
              <div style={{ marginTop: '18px' }}>
                <Button onClick={handleGenerateCalendar} variant="primary" style={{ width: '100%' }} loading={forecasting} disabled={!selectedDistrict || forecasting}>
                  Generate 30-Day Calendar
                </Button>
              </div>
            </Card>

            {forecasting && <Loading message="Generating forecast..." />}

            {calendarPredictions.length > 0 && (
              <Card title="Monthly Risk Calendar">
                <MonthlyCalendar predictions={calendarPredictions} />
              </Card>
            )}

            {!forecasting && calendarPredictions.length === 0 && (
              <div className="glass-card" style={{ textAlign: 'center', padding: '48px 0', color: '#6b7280', fontSize: '15px' }}>
                <div style={{ marginBottom: '12px', display: 'flex', justifyContent: 'center', color: '#4b5563' }}>
                  <Calendar size={48} />
                </div>
                <p style={{ color: '#9ca3af' }}>Select a district and click &ldquo;Generate 30-Day Calendar&rdquo; to view forecast</p>
              </div>
            )}
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}
