import { useState, useMemo } from "react";
import { AlertTriangle, TrendingUp, Target, Calendar } from "lucide-react";
import { ResponsiveContainer } from "recharts";

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

export default function HotspotRanking() {
    const [year, setYear] = useState(2026);
    const [month, setMonth] = useState(1);
    const [loadingHotspots, setLoadingHotspots] = useState(false);
    const [hotspotError, setHotspotError] = useState("");
    const [hotspots, setHotspots] = useState([]);

    const monthLabel = useMemo(() => {
        const m = MONTHS.find((x) => x.v === Number(month));
        return m ? m.label : "";
    }, [month]);

    async function onLoadHotspots() {
        setHotspotError("");
        setHotspots([]);
        setLoadingHotspots(true);

        try {
            const url = `${API_BASE}/hotspots?year=${encodeURIComponent(
                Number(year)
            )}&month=${encodeURIComponent(Number(month))}&top_k=10`;

            const res = await fetch(url);
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || `Request failed (${res.status})`);
            }

            const data = await res.json();
            setHotspots(data.items || []);
        } catch (e) {
            setHotspotError(String(e.message || e));
        } finally {
            setLoadingHotspots(false);
        }
    }

    return (
        <div className="dashboard-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Top 10 Risk Hotspots</h1>
                    <p className="page-subtitle">Identify the most vulnerable locations for poaching this month.</p>
                </div>
                <div className="status-card-mini">
                    <AlertTriangle size={20} />
                    <span>Threat Prioritization</span>
                </div>
            </div>

            <div className="glass-card input-section" style={{ marginBottom: "2rem" }}>
                <div className="input-grid" style={{ gridTemplateColumns: "1fr 1fr auto" }}>
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

                    <button className="predict-btn" onClick={onLoadHotspots} disabled={loadingHotspots} style={{ marginTop: "24px" }}>
                        <TrendingUp size={20} />
                        {loadingHotspots ? "Generating..." : "Generate Ranking"}
                    </button>
                </div>
            </div>

            <div className="glass-card hotspot-section">
                <div className="card-header">
                    <div className="header-left">
                        <Target size={20} />
                        <h2>Hotspot Ranking Results</h2>
                    </div>
                </div>

                <p className="hotspot-subtitle">
                    Ranking for <strong>{year}</strong> • <strong>{monthLabel}</strong>
                </p>

                {hotspotError && (
                    <div className="alert-error">
                        <AlertTriangle size={18} />
                        {hotspotError}
                    </div>
                )}

                {hotspots.length === 0 && !loadingHotspots && (
                    <div className="empty-state-small">
                        <p>Select a time period and click "Generate Ranking" to see the highest-risk locations</p>
                    </div>
                )}

                {hotspots.length > 0 && (
                    <div className="hotspot-table-wrapper">
                        <table className="hotspot-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Region</th>
                                    <th>Location</th>
                                    <th>Risk %</th>
                                    <th>Level</th>
                                    <th>Predicted Offence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {hotspots.map((h) => (
                                    <tr key={`${h.region}-${h.location}-${h.rank}`} className="hotspot-row">
                                        <td>
                                            <span className="rank-badge">{h.rank}</span>
                                        </td>
                                        <td className="region-cell">{h.region}</td>
                                        <td className="location-cell">{h.location}</td>
                                        <td className="risk-cell">
                                            <div className="risk-bar-container">
                                                <div className="risk-bar" style={{ width: `${h.risk_percent}%` }}></div>
                                                <span className="risk-text">{h.risk_percent}%</span>
                                            </div>
                                        </td>
                                        <td>
                                            <RiskBadge level={h.risk_level} />
                                        </td>
                                        <td className="offence-cell">{h.predicted_offence_type}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}
