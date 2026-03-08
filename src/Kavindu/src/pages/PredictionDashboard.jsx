import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, MapPin, Calendar, TrendingUp, Target, Zap } from "lucide-react";
import { RadialBarChart, RadialBar, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from "recharts";

const API_BASE = "http://localhost:8002";

const MONTHS = [
  { v: 1, label: "1 - January" },
  { v: 2, label: "2 - February" },
  { v: 3, label: "3 - March" },
  { v: 4, label: "4 - April" },
  { v: 5, label: "5 - May" },
  { v: 6, label: "6 - June" },
  { v: 7, label: "7 - July" },
  { v: 8, label: "8 - August" },
  { v: 9, label: "9 - September" },
  { v: 10, label: "10 - October" },
  { v: 11, label: "11 - November" },
  { v: 12, label: "12 - December" },
];

function RiskBadge({ level }) {
  const config = {
    High: { className: "risk-badge-high", icon: "🔴" },
    Medium: { className: "risk-badge-medium", icon: "🟡" },
    Low: { className: "risk-badge-low", icon: "🟢" },
  };
  const { className, icon } = config[level] || { className: "risk-badge-low", icon: "⚪" };
  return (
    <span className={`risk-badge ${className}`}>
      <span className="risk-icon">{icon}</span>
      {level || "—"}
    </span>
  );
}

export default function PredictionDashboard() {
  const [regions, setRegions] = useState([]);
  const [locations, setLocations] = useState([]);

  const [region, setRegion] = useState("");
  const [location, setLocation] = useState("");
  const [year, setYear] = useState(2026);
  const [month, setMonth] = useState(1);

  const [loadingPredict, setLoadingPredict] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [result, setResult] = useState(null);

  const monthLabel = useMemo(() => {
    const m = MONTHS.find((x) => x.v === Number(month));
    return m ? m.label : "";
  }, [month]);

  // Prepare chart data
  const riskChartData = useMemo(() => {
    if (!result) return null;
    return [
      {
        name: "Risk Level",
        value: result.risk_percent,
        fill: result.risk_level === "High" ? "#ef4444" : result.risk_level === "Medium" ? "#f59e0b" : "#10b981",
      },
    ];
  }, [result]);

  const offenceDistribution = useMemo(() => {
    if (!result || !result.top3_offence_types) return [];
    return result.top3_offence_types.map((type, idx) => ({
      name: type,
      value: 33 - idx * 8, // Mock distribution
      color: ["#10b981", "#14b8a6", "#06b6d4"][idx] || "#6366f1",
    }));
  }, [result]);

  // Load regions on start
  useEffect(() => {
    fetch(`${API_BASE}/regions`)
      .then((r) => r.json())
      .then((data) => {
        const list = data.regions || [];
        setRegions(list);
        if (list.length > 0) setRegion(list[0]);
      })
      .catch(() => { });
  }, []);

  // Load locations when region changes
  useEffect(() => {
    if (!region) return;
    fetch(`${API_BASE}/locations?region=${encodeURIComponent(region)}`)
      .then((r) => r.json())
      .then((data) => {
        const list = data.locations || [];
        setLocations(list);
        if (list.length > 0) setLocation(list[0]);
      })
      .catch(() => { });
  }, [region]);

  async function onPredict() {
    setPredictError("");
    setResult(null);
    setLoadingPredict(true);

    try {
      const payload = {
        region,
        location,
        year: Number(year),
        month: Number(month),
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setPredictError(String(e.message || e));
    } finally {
      setLoadingPredict(false);
    }
  }

  return (
    <div className="dashboard-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Wildlife Offence Prediction Dashboard</h1>
          <p className="page-subtitle">Advanced AI-powered risk assessment for wildlife conservation</p>
        </div>
        <div className="status-card-mini">
          <Target size={20} />
          <span>Predictive Analysis</span>
        </div>
      </div>

      {/* Input Section */}
      <div className="glass-card input-section">
        <div className="card-header">
          <MapPin size={20} />
          <h2>Location & Time Selection</h2>
        </div>

        <div className="input-grid">
          <div className="input-group">
            <label>
              <span className="label-icon">🌍</span>
              Region
            </label>
            <select value={region} onChange={(e) => setRegion(e.target.value)} className="glass-input">
              {regions.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <label>
              <span className="label-icon">📍</span>
              Location
            </label>
            <select value={location} onChange={(e) => setLocation(e.target.value)} className="glass-input">
              {locations.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <label>
              <Calendar size={16} />
              Year
            </label>
            <input
              type="number"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              min={2000}
              max={2100}
              className="glass-input"
            />
          </div>

          <div className="input-group">
            <label>
              <Calendar size={16} />
              Month
            </label>
            <select value={month} onChange={(e) => setMonth(e.target.value)} className="glass-input">
              {MONTHS.map((m) => (
                <option key={m.v} value={m.v}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button className="predict-btn" onClick={onPredict} disabled={loadingPredict}>
          <Zap size={20} />
          {loadingPredict ? "Analyzing..." : "Run Prediction Analysis"}
        </button>

        {predictError && (
          <div className="alert-error">
            <AlertTriangle size={18} />
            {predictError}
          </div>
        )}
      </div>

      {/* Results Section */}
      {result && (
        <div className="results-grid">
          {/* Risk Score Visualization */}
          <div className="glass-card result-card">
            <div className="card-header">
              <TrendingUp size={20} />
              <h2>Risk Assessment</h2>
            </div>

            <div className="risk-visual">
              <div className="risk-chart">
                <ResponsiveContainer width="100%" height={200}>
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius="60%"
                    outerRadius="90%"
                    barSize={20}
                    data={riskChartData}
                    startAngle={180}
                    endAngle={0}
                  >
                    <RadialBar
                      minAngle={15}
                      background
                      clockWise
                      dataKey="value"
                      cornerRadius={10}
                    />
                  </RadialBarChart>
                </ResponsiveContainer>
                <div className="risk-center-text">
                  <div className="risk-percent">{result.risk_percent}%</div>
                  <RiskBadge level={result.risk_level} />
                </div>
              </div>

              <div className="risk-details">
                <div className="detail-item">
                  <span className="detail-label">Location ID</span>
                  <span className="detail-value">{result.location_id}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Period</span>
                  <span className="detail-value">
                    {year} • {monthLabel}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Offence Type Prediction */}
          <div className="glass-card result-card">
            <div className="card-header">
              <Target size={20} />
              <h2>Offence Type Prediction</h2>
            </div>

            <div className="offence-section">
              <div className="primary-offence">
                <span className="offence-label">Primary Threat</span>
                <div className="offence-type-main">{result.predicted_offence_type}</div>
              </div>

              {offenceDistribution.length > 0 && (
                <div className="offence-chart">
                  <ResponsiveContainer width="100%" height={180}>
                    <PieChart>
                      <Pie
                        data={offenceDistribution}
                        cx="50%"
                        cy="50%"
                        innerRadius={45}
                        outerRadius={70}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {offenceDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}

              <div className="top-threats">
                <span className="threats-label">Top 3 Threat Types</span>
                <div className="threat-chips">
                  {(result.top3_offence_types || []).map((t, idx) => (
                    <div key={t} className="threat-chip" style={{ "--idx": idx }}>
                      <span className="rank">#{idx + 1}</span>
                      {t}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {!result && !loadingPredict && (
        <div className="glass-card empty-state">
          <div className="empty-icon">📊</div>
          <h3>No Prediction Data</h3>
          <p>Select a location and time period, then run the prediction to see AI-powered risk analysis</p>
        </div>
      )}

    </div>
  );
}
