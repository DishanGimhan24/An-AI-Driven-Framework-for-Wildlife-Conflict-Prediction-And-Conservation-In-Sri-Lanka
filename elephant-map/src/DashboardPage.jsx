import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import axios from "axios";

const FEATURES = [
  {
    path: "/map",
    icon: "🗺️",
    title: "Elephant Movement Map",
    desc: "Interactive map of 159 hotspot zones and movement corridors. Filter by district, danger level, elephant count, and active hours. Click corridors to view human-distance zones.",
    color: "#1565C0",
    bgColor: "#E3F2FD",
    border: "#90CAF9",
  },
  {
    path: "/hotspots",
    icon: "🐘",
    title: "Hotspot Zones",
    desc: "Browse all DBSCAN-clustered elephant activity zones. Each hotspot shows sighting count, elephant identities, NDVI, proximity to humans, and safety score.",
    color: "#2E7D32",
    bgColor: "#E8F5E9",
    border: "#A5D6A7",
  },
  {
    path: "/corridors",
    icon: "🛤️",
    title: "Movement Corridors",
    desc: "Explore elephant movement paths between hotspots. See safety scores, usage frequency, active hours, corridor length, and road crossing count per corridor.",
    color: "#6A1B9A",
    bgColor: "#F3E5F5",
    border: "#CE93D8",
  },
  {
    path: "/road-crossings",
    icon: "🚗",
    title: "Road Crossings",
    desc: "Where elephant corridors cross roads. Includes danger score, night crossing ratio, peak season, traffic exposure, and named elephant crossings.",
    color: "#B71C1C",
    bgColor: "#FFEBEE",
    border: "#EF9A9A",
  },
  {
    path: "/predict",
    icon: "🔮",
    title: "Risk Prediction",
    desc: "Click any location in Sri Lanka for an AI-powered conflict risk score using XGBoost ML. Get elephant presence probability, nearest hotspot & corridor details, and safety recommendations.",
    color: "#E65100",
    bgColor: "#FFF3E0",
    border: "#FFCC80",
  },
];

function StatCard({ icon, label, value, sub, color }) {
  return (
    <div style={{
      backgroundColor: "white",
      borderRadius: "12px",
      padding: "20px 24px",
      boxShadow: "0 2px 12px rgba(0,0,0,0.08)",
      borderLeft: `4px solid ${color}`,
      minWidth: 0,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <span style={{ fontSize: "32px" }}>{icon}</span>
        <div>
          <div style={{ fontSize: "28px", fontWeight: "700", color, lineHeight: 1.1 }}>{value ?? "—"}</div>
          <div style={{ fontSize: "13px", fontWeight: "600", color: "#333", marginTop: "2px" }}>{label}</div>
          {sub && <div style={{ fontSize: "11px", color: "#888", marginTop: "2px" }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function DangerBar({ label, count, total, color }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div style={{ marginBottom: "10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
        <span style={{ fontSize: "13px", fontWeight: "600", color: "#444" }}>{label}</span>
        <span style={{ fontSize: "13px", fontWeight: "700", color }}>{count} ({pct.toFixed(0)}%)</span>
      </div>
      <div style={{ backgroundColor: "#f0f0f0", borderRadius: "6px", height: "10px", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", backgroundColor: color, borderRadius: "6px", transition: "width 0.6s ease" }} />
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState(false);

  useEffect(() => {
    Promise.all([
      axios.get("http://localhost:8000/health"),
      axios.get("http://localhost:8000/road-crossings/summary"),
    ])
      .then(([healthRes, summaryRes]) => {
        setStats(healthRes.data);
        setSummary(summaryRes.data);
        setLoading(false);
      })
      .catch(() => {
        setApiError(true);
        setLoading(false);
      });
  }, []);

  const totalNodes     = stats?.corridor_network?.total_nodes ?? 0;
  const totalCorridors = stats?.corridor_network?.total_corridors ?? 0;
  const totalCrossings = stats?.corridor_network?.total_road_crossings ?? 0;
  const dangerCounts   = summary?.by_danger_level ?? {};
  const top5           = summary?.top_5_dangerous ?? [];

  const dangerBadge = (level) => {
    const map = { High: { color: "#d32f2f", bg: "#ffebee", icon: "🔴" }, Medium: { color: "#f57c00", bg: "#fff3e0", icon: "🟠" }, Low: { color: "#388e3c", bg: "#e8f5e9", icon: "🟡" } };
    return map[level] ?? { color: "#757575", bg: "#f5f5f5", icon: "⚪" };
  };

  return (
    <div style={{ minHeight: "calc(100vh - 60px)", backgroundColor: "#f4f6fb", paddingBottom: "40px" }}>

      {/* Hero */}
      <div style={{
        background: "linear-gradient(135deg, #0d1b5e 0%, #1a237e 50%, #283593 100%)",
        color: "white",
        padding: "48px 40px 40px",
        textAlign: "center",
      }}>
        <div style={{ fontSize: "56px", marginBottom: "12px" }}>🐘</div>
        <h1 style={{ margin: "0 0 10px 0", fontSize: "30px", fontWeight: "700", letterSpacing: "0.3px" }}>
          AI-Driven Wildlife Conflict Prediction
        </h1>
        <p style={{ margin: "0 auto", maxWidth: "600px", fontSize: "15px", opacity: 0.8, lineHeight: 1.6 }}>
          An intelligent framework for elephant movement analysis, conflict hotspot detection, corridor mapping, and road-crossing risk assessment across Sri Lanka.
        </p>
        {apiError && (
          <div style={{ marginTop: "16px", display: "inline-block", backgroundColor: "rgba(255,80,80,0.25)", border: "1px solid rgba(255,80,80,0.5)", borderRadius: "8px", padding: "8px 20px", fontSize: "13px" }}>
            ⚠️ API server not reachable — start it with <code style={{ backgroundColor: "rgba(255,255,255,0.15)", padding: "2px 6px", borderRadius: "4px" }}>python run.py</code>
          </div>
        )}
      </div>

      <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 24px" }}>

        {/* Stat Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "16px", marginTop: "-24px", marginBottom: "32px" }}>
          <StatCard icon="🐘" label="Hotspot Zones" value={totalNodes} sub="DBSCAN clusters" color="#1565C0" />
          <StatCard icon="🛤️" label="Corridors" value={totalCorridors} sub="Movement paths" color="#6A1B9A" />
          <StatCard icon="🚗" label="Road Crossings" value={totalCrossings} sub="Corridor ✕ road intersections" color="#B71C1C" />
          <StatCard icon="⚠️" label="High Danger" value={dangerCounts.High ?? 0} sub="Critical road crossings" color="#d32f2f" />
        </div>

        {/* Main content grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "24px", alignItems: "start" }}>

          {/* Feature cards */}
          <div>
            <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#1a237e", margin: "0 0 16px 0" }}>
              System Features
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              {FEATURES.map(f => (
                <Link key={f.path} to={f.path} style={{ textDecoration: "none" }}>
                  <div style={{
                    backgroundColor: "white",
                    borderRadius: "12px",
                    padding: "20px",
                    boxShadow: "0 2px 10px rgba(0,0,0,0.07)",
                    border: `1px solid ${f.border}`,
                    transition: "transform 0.2s, box-shadow 0.2s",
                    cursor: "pointer",
                    height: "100%",
                    boxSizing: "border-box"
                  }}
                    onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-3px)"; e.currentTarget.style.boxShadow = "0 6px 20px rgba(0,0,0,0.12)"; }}
                    onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)"; e.currentTarget.style.boxShadow = "0 2px 10px rgba(0,0,0,0.07)"; }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "10px" }}>
                      <div style={{ width: "40px", height: "40px", backgroundColor: f.bgColor, borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "20px", flexShrink: 0 }}>
                        {f.icon}
                      </div>
                      <span style={{ fontSize: "14px", fontWeight: "700", color: f.color }}>{f.title}</span>
                    </div>
                    <p style={{ margin: 0, fontSize: "12px", color: "#666", lineHeight: "1.6" }}>{f.desc}</p>
                    <div style={{ marginTop: "12px", fontSize: "12px", color: f.color, fontWeight: "600", display: "flex", alignItems: "center", gap: "4px" }}>
                      Open page →
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>

          {/* Right column */}
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {/* Danger breakdown */}
            <div style={{ backgroundColor: "white", borderRadius: "12px", padding: "20px", boxShadow: "0 2px 10px rgba(0,0,0,0.07)" }}>
              <h3 style={{ margin: "0 0 16px 0", fontSize: "15px", fontWeight: "700", color: "#333" }}>
                🚗 Road Crossing Danger
              </h3>
              <DangerBar label="🔴 High"   count={dangerCounts.High   ?? 0} total={totalCrossings} color="#d32f2f" />
              <DangerBar label="🟠 Medium" count={dangerCounts.Medium ?? 0} total={totalCrossings} color="#f57c00" />
              <DangerBar label="🟡 Low"    count={dangerCounts.Low    ?? 0} total={totalCrossings} color="#fbc02d" />
              <Link to="/road-crossings" style={{ display: "block", marginTop: "14px", textAlign: "center", backgroundColor: "#B71C1C", color: "white", padding: "9px", borderRadius: "7px", fontSize: "13px", fontWeight: "600", textDecoration: "none" }}>
                View All Crossings →
              </Link>
            </div>

            {/* Top 5 dangerous */}
            {top5.length > 0 && (
              <div style={{ backgroundColor: "white", borderRadius: "12px", padding: "20px", boxShadow: "0 2px 10px rgba(0,0,0,0.07)" }}>
                <h3 style={{ margin: "0 0 14px 0", fontSize: "15px", fontWeight: "700", color: "#333" }}>
                  ⚠️ Top 5 Dangerous Crossings
                </h3>
                {top5.map((c, i) => {
                  const b = dangerBadge(c.danger_level);
                  return (
                    <div key={i} style={{ padding: "10px", backgroundColor: i % 2 === 0 ? "#fafafa" : "white", borderRadius: "6px", marginBottom: "6px", fontSize: "12px", borderLeft: `3px solid ${b.color}` }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                        <span style={{ fontWeight: "600", color: "#333" }}>Corridor {c.corridor_id}</span>
                        <span style={{ backgroundColor: b.bg, color: b.color, padding: "1px 7px", borderRadius: "10px", fontWeight: "700", fontSize: "11px" }}>{b.icon} {c.danger_level}</span>
                      </div>
                      <div style={{ color: "#666", lineHeight: 1.6 }}>
                        <span>{c.road_type} · Score: <strong style={{ color: b.color }}>{c.danger_score.toFixed(0)}</strong></span><br />
                        <span>🌙 Night: {(c.night_ratio * 100).toFixed(0)}% · 🌿 Season: {c.peak_season}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* System info */}
            <div style={{ backgroundColor: "white", borderRadius: "12px", padding: "20px", boxShadow: "0 2px 10px rgba(0,0,0,0.07)" }}>
              <h3 style={{ margin: "0 0 12px 0", fontSize: "15px", fontWeight: "700", color: "#333" }}>
                ⚙️ System Info
              </h3>
              <div style={{ fontSize: "12px", color: "#555", lineHeight: 2 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>ML Model</span><span style={{ fontWeight: "600" }}>XGBoost</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Clustering</span><span style={{ fontWeight: "600" }}>DBSCAN · 0.5 km radius</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Distance Metric</span><span style={{ fontWeight: "600" }}>Haversine</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Study Region</span><span style={{ fontWeight: "600" }}>Sri Lanka</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>API Status</span>
                  <span style={{ fontWeight: "600", color: apiError ? "#d32f2f" : "#388e3c" }}>
                    {loading ? "Checking..." : apiError ? "Offline" : "✅ Online"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
