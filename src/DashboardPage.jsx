import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { DISHAN_API } from "./apiConfig";
import { AlertTriangle, Cpu, Car, Activity } from "lucide-react";

const FEATURES = [
  {
    path: "/map",
    icon: "🗺️",
    title: "Elephant Movement Map",
    desc: "Interactive map of 159 hotspot zones and movement corridors. Filter by district, danger level, elephant count, and active hours. Click corridors to view human-distance zones.",
    accent: "#34d399",
  },
  {
    path: "/hotspots",
    icon: "🐘",
    title: "Hotspot Zones",
    desc: "Browse all DBSCAN-clustered elephant activity zones. Each hotspot shows sighting count, elephant identities, NDVI, proximity to humans, and safety score.",
    accent: "#10b981",
  },
  {
    path: "/corridors",
    icon: "🛤️",
    title: "Movement Corridors",
    desc: "Explore elephant movement paths between hotspots. See safety scores, usage frequency, active hours, corridor length, and road crossing count per corridor.",
    accent: "#a78bfa",
  },
  {
    path: "/road-crossings",
    icon: "🚗",
    title: "Road Crossings",
    desc: "Where elephant corridors cross roads. Includes danger score, night crossing ratio, peak season, traffic exposure, and named elephant crossings.",
    accent: "#f87171",
  },
  {
    path: "/predict",
    icon: "🔮",
    title: "Risk Prediction",
    desc: "Click any location in Sri Lanka for an AI-powered conflict risk score using XGBoost ML. Get elephant presence probability, nearest hotspot & corridor details, and safety recommendations.",
    accent: "#fb923c",
  },
  {
    path: "/risk-dashboard",
    icon: "📊",
    title: "Risk Analytics Dashboard",
    desc: "CSV-driven KPI board showing high/medium/low risk point counts, cluster stats, vehicle type breakdown, and top dangerous districts from ML output data.",
    accent: "#34d399",
  },
  {
    path: "/risk-map",
    icon: "🗾",
    title: "Collision Risk Map",
    desc: "Interactive map of all wildlife-vehicle collision incidents loaded from CSV. Filter by vehicle type, risk level, and toggle DBSCAN cluster centers. Switch between basemaps.",
    accent: "#60a5fa",
  },
  {
    path: "/risk-prediction",
    icon: "🧮",
    title: "District Risk Predictor",
    desc: "Select a district and enter environmental parameters (rainfall, NDVI, water/forest distances) to get an AI-predicted wildlife collision risk score and level.",
    accent: "#818cf8",
  },
  {
    path: "/avc-home",
    icon: "🌿",
    title: "AVC Platform Home",
    desc: "Overview of the AI-Smart Animal Vehicle Collision Predictor platform — mission, stats, and feature highlights for the Sri Lanka wildlife safety initiative.",
    accent: "#34d399",
  },
  // ── Tharushi (ELESAFE) ──────────────────────────────────────
  {
    path: "/tharushi/login",
    icon: "🛡️",
    title: "ELESAFE Login",
    desc: "Sign in to the ELESAFE wildlife conflict prediction platform to access personalised risk dashboards, forecasts, and historical analysis.",
    accent: "#f472b6",
  },
  {
    path: "/tharushi/dashboard",
    icon: "📈",
    title: "ELESAFE Dashboard",
    desc: "Real-time overview of wildlife conflict risk across cities — stat cards, heat maps, risk trends, and forecast summaries powered by the ELESAFE AI model.",
    accent: "#e879f9",
  },
  {
    path: "/tharushi/predict",
    icon: "🧠",
    title: "ELESAFE Risk Prediction",
    desc: "Select a city and date range to run ELESAFE's ML risk prediction. View risk score, level badge, contributing factors, and confidence breakdown.",
    accent: "#c084fc",
  },
  {
    path: "/tharushi/map-calendar",
    icon: "🗓️",
    title: "ELESAFE Map Calendar",
    desc: "Visualise daily wildlife conflict risk on an interactive map calendar. Browse predicted risk levels by date across all monitored cities in Sri Lanka.",
    accent: "#a78bfa",
  },
  {
    path: "/tharushi/historical",
    icon: "📜",
    title: "ELESAFE Historical",
    desc: "Explore historical wildlife conflict records with trend charts, district breakdowns, and seasonal pattern analysis from the ELESAFE dataset.",
    accent: "#818cf8",
  },
  // ── Kavindu (Wildlife Command Center) ───────────────────────
  {
    path: "/kavindu",
    icon: "🏠",
    title: "Command Center Home",
    desc: "Landing page for the Wildlife Command Center — overview of the officer management, hotspot ranking, prediction dashboard, and reporting system.",
    accent: "#38bdf8",
  },
  {
    path: "/kavindu/dashboard",
    icon: "📡",
    title: "Prediction Dashboard",
    desc: "Unified prediction dashboard for wildlife officers — view ML model outputs, district risk rankings, and real-time alert summaries across Sri Lanka.",
    accent: "#0ea5e9",
  },
  {
    path: "/kavindu/hotspots",
    icon: "🔥",
    title: "Hotspot Ranking",
    desc: "Ranked list of wildlife conflict hotspots with risk scores, incident counts, and district details. Supports officer patrol planning and resource allocation.",
    accent: "#f97316",
  },
  {
    path: "/kavindu/report",
    icon: "📋",
    title: "Incident Report",
    desc: "Submit and review wildlife conflict incident reports. Capture location, animal type, severity, and officer notes for centralised record keeping.",
    accent: "#fb923c",
  },
  {
    path: "/kavindu/officer/login",
    icon: "👮",
    title: "Officer Login",
    desc: "Secure login portal for wildlife field officers to access the Command Center dashboard, assignments, and incident reporting tools.",
    accent: "#34d399",
  },
  {
    path: "/kavindu/admin",
    icon: "⚙️",
    title: "Admin Panel",
    desc: "Administrator panel for managing officers, system settings, user roles, and reviewing all submitted incident reports across the Command Center.",
    accent: "#94a3b8",
  },
];

const G = {
  bg: "#0a0e14",
  card: "rgba(255,255,255,0.06)",
  border: "rgba(255,255,255,0.10)",
  shadow: "0 8px 32px rgba(0,0,0,0.4)",
  textPrimary: "#f9fafb",
  textSecondary: "rgba(255,255,255,0.65)",
  textMuted: "rgba(255,255,255,0.38)",
  emerald: "#10b981",
  emeraldLight: "#34d399",
};

function StatCard({ icon, label, value, sub, color }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.06)",
      backdropFilter: "blur(20px)",
      WebkitBackdropFilter: "blur(20px)",
      borderRadius: "16px",
      padding: "20px 22px",
      border: "1px solid rgba(255,255,255,0.10)",
      boxShadow: "0 8px 32px rgba(0,0,0,0.35)",
      borderLeft: `3px solid ${color}`,
      minWidth: 0,
      transition: "transform 0.2s, box-shadow 0.2s",
    }}
      onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-3px)"; e.currentTarget.style.boxShadow = "0 12px 40px rgba(0,0,0,0.5)"; }}
      onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)"; e.currentTarget.style.boxShadow = "0 8px 32px rgba(0,0,0,0.35)"; }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
        <span style={{ fontSize: "30px" }}>{icon}</span>
        <div>
          <div style={{ fontSize: "28px", fontWeight: "800", color, lineHeight: 1.1, letterSpacing: "-1px" }}>{value ?? "—"}</div>
          <div style={{ fontSize: "13px", fontWeight: "600", color: G.textPrimary, marginTop: "3px" }}>{label}</div>
          {sub && <div style={{ fontSize: "11px", color: G.textMuted, marginTop: "2px" }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function DangerBar({ label, count, total, color }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div style={{ marginBottom: "14px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
        <span style={{ fontSize: "13px", fontWeight: "600", color: G.textSecondary }}>{label}</span>
        <span style={{ fontSize: "13px", fontWeight: "700", color }}>{count} ({pct.toFixed(0)}%)</span>
      </div>
      <div style={{ background: "rgba(255,255,255,0.08)", borderRadius: "8px", height: "8px", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: `linear-gradient(90deg, ${color}, ${color}99)`, borderRadius: "8px", transition: "width 0.6s ease" }} />
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
      axios.get(`${DISHAN_API}/health`),
      axios.get(`${DISHAN_API}/road-crossings/summary`),
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

  const totalNodes     = stats?.corridor_network?.total_nodes ?? stats?.total_nodes ?? 159;
  const totalCorridors = stats?.corridor_network?.total_corridors ?? stats?.total_corridors ?? 283;
  const totalCrossings = summary?.total ?? stats?.corridor_network?.total_road_crossings ?? 0;
  const dangerCounts   = summary?.by_danger_level ?? {};
  const top5           = summary?.top_5_dangerous ?? [];

  const dangerBadge = (level) => {
    const map = {
      High:   { color: "#ef4444", bg: "rgba(239,68,68,0.15)",   icon: "🔴" },
      Medium: { color: "#f59e0b", bg: "rgba(245,158,11,0.15)",  icon: "🟠" },
      Low:    { color: "#10b981", bg: "rgba(16,185,129,0.15)",  icon: "🟡" },
    };
    return map[level] ?? { color: "#6b7280", bg: "rgba(107,114,128,0.15)", icon: "⚪" };
  };

  return (
    <div style={{ minHeight: "calc(100vh - 60px)", background: `linear-gradient(135deg, ${G.bg} 0%, #064e3b22 50%, ${G.bg} 100%)`, paddingBottom: "48px", fontFamily: "'Inter', -apple-system, sans-serif" }}>

      {/* Hero */}
      <div style={{
        background: "linear-gradient(135deg, #0a0e14 0%, #064e3b 50%, #0a0e14 100%)",
        padding: "56px 40px 48px",
        textAlign: "center",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        position: "relative",
        overflow: "hidden",
      }}>
        {/* Background glow */}
        <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", width: "600px", height: "300px", background: "radial-gradient(ellipse, rgba(16,185,129,0.12) 0%, transparent 70%)", pointerEvents: "none" }} />
        <div style={{ position: "relative", zIndex: 1 }}>
          <div style={{ fontSize: "64px", marginBottom: "16px", animation: "none" }}>🐘</div>
          <h1 style={{
            margin: "0 0 12px 0",
            fontSize: "32px",
            fontWeight: "800",
            letterSpacing: "-0.5px",
            background: "linear-gradient(135deg, #ffffff 0%, #34d399 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
          }}>
            AI-Driven Wildlife Conflict Prediction
          </h1>
          <p style={{ margin: "0 auto", maxWidth: "580px", fontSize: "15px", color: "rgba(255,255,255,0.6)", lineHeight: 1.7 }}>
            An intelligent framework for elephant movement analysis, conflict hotspot detection, corridor mapping, and road-crossing risk assessment across Sri Lanka.
          </p>
          {apiError && (
            <div style={{ marginTop: "20px", display: "inline-block", background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "10px", padding: "10px 22px", fontSize: "13px", color: "#f87171" }}>
              ⚠️ API server not reachable — start it with{" "}
              <code style={{ background: "rgba(255,255,255,0.1)", padding: "2px 6px", borderRadius: "4px" }}>python run.py</code>
            </div>
          )}
        </div>
      </div>

      <div style={{ maxWidth: "1240px", margin: "0 auto", padding: "0 24px" }}>

        {/* Stat Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "16px", marginTop: "-28px", marginBottom: "36px", position: "relative", zIndex: 2 }}>
          <StatCard icon="🐘" label="Hotspot Zones"  value={totalNodes}     sub="DBSCAN clusters"               color="#34d399" />
          <StatCard icon="🛤️" label="Corridors"      value={totalCorridors} sub="Movement paths"                color="#a78bfa" />
          <StatCard icon="🚗" label="Road Crossings" value={totalCrossings} sub="Corridor × road intersections" color="#60a5fa" />
          <StatCard icon="⚠️" label="High Danger"    value={dangerCounts.High ?? 0} sub="Critical road crossings" color="#ef4444" />
        </div>

        {/* Main content grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "24px", alignItems: "start" }}>

          {/* Feature cards */}
          <div>
            <h2 style={{ fontSize: "17px", fontWeight: "700", color: G.textPrimary, margin: "0 0 18px 0", display: "flex", alignItems: "center", gap: "8px" }}>
              <Activity size={18} color="#10b981" />
              System Features
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
              {FEATURES.map(f => (
                <Link key={f.path} to={f.path} style={{ textDecoration: "none" }}>
                  <div style={{
                    background: G.card,
                    backdropFilter: "blur(20px)",
                    WebkitBackdropFilter: "blur(20px)",
                    borderRadius: "16px",
                    padding: "20px",
                    border: `1px solid ${G.border}`,
                    boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
                    transition: "transform 0.2s, box-shadow 0.2s, border-color 0.2s",
                    cursor: "pointer",
                    height: "100%",
                    boxSizing: "border-box",
                    borderTop: `2px solid ${f.accent}`,
                  }}
                    onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-4px)"; e.currentTarget.style.boxShadow = "0 12px 36px rgba(0,0,0,0.5)"; e.currentTarget.style.borderColor = f.accent; }}
                    onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)"; e.currentTarget.style.boxShadow = "0 4px 20px rgba(0,0,0,0.3)"; e.currentTarget.style.borderColor = G.border; }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "10px" }}>
                      <div style={{ width: "38px", height: "38px", background: `${f.accent}18`, borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "18px", flexShrink: 0, border: `1px solid ${f.accent}30` }}>
                        {f.icon}
                      </div>
                      <span style={{ fontSize: "13px", fontWeight: "700", color: f.accent }}>{f.title}</span>
                    </div>
                    <p style={{ margin: 0, fontSize: "12px", color: G.textMuted, lineHeight: "1.65" }}>{f.desc}</p>
                    <div style={{ marginTop: "12px", fontSize: "12px", color: f.accent, fontWeight: "600", display: "flex", alignItems: "center", gap: "4px" }}>
                      Open page →
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>

          {/* Right column */}
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

            {/* Danger breakdown */}
            <div style={{
              background: G.card,
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              borderRadius: "16px",
              padding: "22px",
              border: `1px solid ${G.border}`,
              boxShadow: "0 8px 32px rgba(0,0,0,0.35)",
            }}>
              <h3 style={{ margin: "0 0 18px 0", fontSize: "14px", fontWeight: "700", color: G.textPrimary, display: "flex", alignItems: "center", gap: "7px" }}>
                <Car size={15} color="#ef4444" />
                Road Crossing Danger
              </h3>
              <DangerBar label="🔴 High"   count={dangerCounts.High   ?? 0} total={totalCrossings} color="#ef4444" />
              <DangerBar label="🟠 Medium" count={dangerCounts.Medium ?? 0} total={totalCrossings} color="#f59e0b" />
              <DangerBar label="🟡 Low"    count={dangerCounts.Low    ?? 0} total={totalCrossings} color="#fbbf24" />
              <Link to="/road-crossings" style={{
                display: "block", marginTop: "16px", textAlign: "center",
                background: "linear-gradient(135deg, #991b1b, #b91c1c)",
                color: "white", padding: "10px", borderRadius: "10px",
                fontSize: "13px", fontWeight: "600", textDecoration: "none",
                boxShadow: "0 4px 12px rgba(239,68,68,0.25)",
                transition: "opacity 0.2s",
              }}>
                View All Crossings →
              </Link>
            </div>

            {/* Top 5 dangerous */}
            {top5.length > 0 && (
              <div style={{
                background: G.card,
                backdropFilter: "blur(20px)",
                WebkitBackdropFilter: "blur(20px)",
                borderRadius: "16px",
                padding: "22px",
                border: `1px solid ${G.border}`,
                boxShadow: "0 8px 32px rgba(0,0,0,0.35)",
              }}>
                <h3 style={{ margin: "0 0 16px 0", fontSize: "14px", fontWeight: "700", color: G.textPrimary, display: "flex", alignItems: "center", gap: "7px" }}>
                  <AlertTriangle size={15} color="#f59e0b" />
                  Top 5 Dangerous Crossings
                </h3>
                {top5.map((c, i) => {
                  const b = dangerBadge(c.danger_level);
                  return (
                    <div key={i} style={{
                      padding: "12px",
                      background: "rgba(255,255,255,0.04)",
                      borderRadius: "10px",
                      marginBottom: "8px",
                      fontSize: "12px",
                      borderLeft: `3px solid ${b.color}`,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                        <span style={{ fontWeight: "600", color: G.textPrimary }}>Corridor {c.corridor_id}</span>
                        <span style={{ background: b.bg, color: b.color, padding: "2px 8px", borderRadius: "10px", fontWeight: "700", fontSize: "11px", border: `1px solid ${b.color}30` }}>
                          {b.icon} {c.danger_level}
                        </span>
                      </div>
                      <div style={{ color: G.textMuted, lineHeight: 1.6 }}>
                        <span>{c.road_type} · Score: <strong style={{ color: b.color }}>{c.danger_score.toFixed(0)}</strong></span><br />
                        <span>🌙 Night: {(c.night_ratio * 100).toFixed(0)}% · 🌿 {c.peak_season}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* System info */}
            <div style={{
              background: G.card,
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              borderRadius: "16px",
              padding: "22px",
              border: `1px solid ${G.border}`,
              boxShadow: "0 8px 32px rgba(0,0,0,0.35)",
            }}>
              <h3 style={{ margin: "0 0 14px 0", fontSize: "14px", fontWeight: "700", color: G.textPrimary, display: "flex", alignItems: "center", gap: "7px" }}>
                <Cpu size={15} color="#34d399" />
                System Info
              </h3>
              <div style={{ fontSize: "12px", color: G.textSecondary, lineHeight: 1 }}>
                {[
                  ["ML Model",         "XGBoost"],
                  ["Clustering",       "DBSCAN · 0.5 km radius"],
                  ["Distance Metric",  "Haversine"],
                  ["Study Region",     "Sri Lanka"],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "9px 0", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                    <span>{k}</span>
                    <span style={{ fontWeight: "600", color: G.textPrimary }}>{v}</span>
                  </div>
                ))}
                <div style={{ display: "flex", justifyContent: "space-between", padding: "9px 0" }}>
                  <span>API Status</span>
                  <span style={{ fontWeight: "600", color: apiError ? "#ef4444" : "#10b981" }}>
                    {loading ? "Checking…" : apiError ? "⚠️ Offline" : "✅ Online"}
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
