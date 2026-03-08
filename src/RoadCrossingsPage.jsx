import { useState, useEffect, useMemo } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { DISHAN_API } from "./apiConfig";
import { Search, Map, RefreshCw } from "lucide-react";

const PAGE_SIZE = 25;

const SEASONS    = ["All", "Dry", "Inter-Monsoon", "Southwest-Monsoon", "Inter-Monsoon-2"];
const ROAD_TYPES = ["All", "Major Road", "Secondary Road", "Minor Road"];
const TIME_LABELS = ["All", "Night", "Early Morning", "Midday", "Afternoon", "Evening"];

const dangerColor = (lvl) => lvl === "High" ? "#ef4444" : lvl === "Medium" ? "#f59e0b" : "#fbbf24";
const dangerBg    = (lvl) => lvl === "High" ? "rgba(239,68,68,0.15)" : lvl === "Medium" ? "rgba(245,158,11,0.15)" : "rgba(251,191,36,0.15)";
const dangerIcon  = (lvl) => lvl === "High" ? "🔴" : lvl === "Medium" ? "🟠" : "🟡";

const mkIcon = (lvl) => L.divIcon({
  html: `<div style="font-size:16px;text-align:center;line-height:1;filter:drop-shadow(0 1px 4px rgba(0,0,0,0.7))">${dangerIcon(lvl)}</div>`,
  className: "",
  iconSize: [20, 20],
  iconAnchor: [10, 10],
  popupAnchor: [0, -12]
});

const G = {
  bg: "#0a0e14",
  card: "rgba(255,255,255,0.06)",
  border: "rgba(255,255,255,0.10)",
  inputBg: "rgba(255,255,255,0.07)",
  textPrimary: "#f9fafb",
  textSecondary: "rgba(255,255,255,0.65)",
  textMuted: "rgba(255,255,255,0.38)",
};

export default function RoadCrossingsPage() {
  const [crossings, setCrossings]       = useState([]);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);
  const [summary, setSummary]           = useState(null);
  const [search, setSearch]             = useState("");
  const [dangerFilter, setDanger]       = useState("All");
  const [roadTypeFilter, setRoadType]   = useState("All");
  const [seasonFilter, setSeason]       = useState("All");
  const [timeFilter, setTime]           = useState("All");
  const [sortBy, setSortBy]             = useState("danger_desc");
  const [page, setPage]                 = useState(0);
  const [mapVisible, setMapVisible]     = useState(true);

  useEffect(() => {
    Promise.all([
      axios.get(`${DISHAN_API}/road-crossings`),
      axios.get(`${DISHAN_API}/road-crossings/summary`),
    ])
      .then(([rcRes, sumRes]) => {
        setCrossings(rcRes.data);
        setSummary(sumRes.data);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const filtered = useMemo(() => crossings.filter(c => {
    if (dangerFilter !== "All" && c.danger_level !== dangerFilter) return false;
    if (roadTypeFilter !== "All" && c.road_type !== roadTypeFilter) return false;
    if (seasonFilter !== "All" && c.peak_season !== seasonFilter) return false;
    if (timeFilter !== "All" && c.peak_time_of_day !== timeFilter) return false;
    if (search) {
      const q = search.toLowerCase();
      if (!c.crossing_id.toLowerCase().includes(q) && !c.corridor_id.toLowerCase().includes(q)) return false;
    }
    return true;
  }), [crossings, dangerFilter, roadTypeFilter, seasonFilter, timeFilter, search]);

  const sorted = useMemo(() => [...filtered].sort((a, b) => {
    if (sortBy === "danger_desc") return b.danger_score - a.danger_score;
    if (sortBy === "danger_asc")  return a.danger_score - b.danger_score;
    if (sortBy === "night_desc")  return b.night_ratio - a.night_ratio;
    if (sortBy === "crossings")   return b.crossing_count - a.crossing_count;
    if (sortBy === "elephants")   return b.elephant_count - a.elephant_count;
    return 0;
  }), [filtered, sortBy]);

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
  const paged = sorted.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

  const mapMarkers = useMemo(() =>
    sorted.filter(c => c.danger_level !== "Low").slice(0, 400),
    [sorted]
  );

  const selectStyle = {
    padding: "8px 12px", fontSize: "13px", borderRadius: "10px",
    border: `1px solid ${G.border}`, background: G.inputBg,
    color: G.textPrimary, cursor: "pointer", outline: "none",
  };

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", background: G.bg, color: G.textSecondary, fontSize: "16px", fontFamily: "'Inter', sans-serif" }}>
      <RefreshCw size={20} style={{ marginRight: "12px", animation: "spin 1s linear infinite" }} />
      Loading road crossings…
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
  if (error) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", background: G.bg, color: "#ef4444", fontFamily: "'Inter', sans-serif" }}>
      ⚠️ {error}. Make sure the API server is running.
    </div>
  );

  const dnCounts = summary?.by_danger_level ?? {};
  const rtCounts = summary?.by_road_type ?? {};

  return (
    <div style={{ minHeight: "calc(100vh - 60px)", background: G.bg, fontFamily: "'Inter', -apple-system, sans-serif" }}>

      {/* Page header */}
      <div style={{
        background: "linear-gradient(135deg, #0a0e14 0%, #7f1d1d 50%, #0a0e14 100%)",
        padding: "28px 32px 24px",
        borderBottom: `1px solid ${G.border}`,
      }}>
        <h1 style={{
          margin: "0 0 6px 0", fontSize: "24px", fontWeight: "800", letterSpacing: "-0.5px",
          background: "linear-gradient(135deg, #ffffff 0%, #fca5a5 100%)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text",
        }}>
          🚗 Road Crossing Analysis
        </h1>
        <p style={{ margin: 0, fontSize: "13px", color: G.textMuted }}>
          {crossings.length.toLocaleString()} crossings where elephant corridors intersect roads · Showing {sorted.length} filtered
        </p>
      </div>

      {/* Summary row */}
      <div style={{ padding: "16px 32px", display: "flex", gap: "16px", flexWrap: "wrap", alignItems: "center", borderBottom: `1px solid ${G.border}` }}>
        <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
          {[
            { label: "🔴 High",   count: dnCounts.High   ?? 0, color: "#ef4444" },
            { label: "🟠 Medium", count: dnCounts.Medium ?? 0, color: "#f59e0b" },
            { label: "🟡 Low",    count: dnCounts.Low    ?? 0, color: "#fbbf24" },
          ].map(d => (
            <div key={d.label} style={{ background: G.card, backdropFilter: "blur(20px)", borderRadius: "12px", padding: "10px 18px", border: `1px solid ${G.border}`, textAlign: "center" }}>
              <div style={{ fontSize: "20px", fontWeight: "800", color: d.color, letterSpacing: "-1px" }}>{d.count.toLocaleString()}</div>
              <div style={{ fontSize: "11px", color: G.textMuted, marginTop: "2px" }}>{d.label} Danger</div>
            </div>
          ))}
        </div>
        <div style={{ width: "1px", background: G.border, height: "36px" }} />
        <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
          {Object.entries(rtCounts).map(([type, cnt]) => (
            <div key={type} style={{ background: G.card, backdropFilter: "blur(20px)", borderRadius: "12px", padding: "10px 16px", border: `1px solid ${G.border}`, textAlign: "center" }}>
              <div style={{ fontSize: "18px", fontWeight: "700", color: G.textPrimary }}>{cnt.toLocaleString()}</div>
              <div style={{ fontSize: "11px", color: G.textMuted, marginTop: "2px" }}>{type}</div>
            </div>
          ))}
        </div>
        <div style={{ marginLeft: "auto" }}>
          <button onClick={() => setMapVisible(v => !v)} style={{
            padding: "8px 16px", borderRadius: "10px",
            border: `1px solid ${mapVisible ? "#ef444460" : G.border}`,
            background: mapVisible ? "rgba(239,68,68,0.15)" : G.inputBg,
            color: mapVisible ? "#f87171" : G.textSecondary,
            fontSize: "12px", fontWeight: "600", cursor: "pointer",
            display: "flex", alignItems: "center", gap: "6px",
            backdropFilter: "blur(10px)",
          }}>
            <Map size={14} />
            {mapVisible ? "Hide Map" : "Show Map"}
          </button>
        </div>
      </div>

      {/* Mini map */}
      {mapVisible && (
        <div style={{ height: "300px", borderBottom: `1px solid ${G.border}` }}>
          <MapContainer center={[7.8731, 80.7718]} zoom={8} style={{ height: "100%", width: "100%" }} zoomControl={true}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            {mapMarkers.map((m, i) => (
              <Marker key={i} position={[m.crossing_lat, m.crossing_lon]} icon={mkIcon(m.danger_level)}>
                <Popup>
                  <div style={{ fontSize: "12px", lineHeight: 1.7, minWidth: "180px" }}>
                    <strong style={{ color: dangerColor(m.danger_level) }}>{m.danger_level} Danger · {m.danger_score.toFixed(0)}</strong><br />
                    Corridor: {m.corridor_id}<br />
                    Road: {m.road_type}<br />
                    🌙 Night: {(m.night_ratio * 100).toFixed(0)}%<br />
                    Peak: {m.peak_season} / {m.peak_time_of_day}
                  </div>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
      )}

      {/* Filter bar */}
      <div style={{ padding: "14px 32px", display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center", borderBottom: `1px solid ${G.border}`, background: "rgba(255,255,255,0.02)" }}>
        <div style={{ position: "relative" }}>
          <Search size={14} style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", color: G.textMuted }} />
          <input
            placeholder="Search crossing or corridor ID…"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            style={{ ...selectStyle, paddingLeft: "34px", minWidth: "230px" }}
          />
        </div>
        <select value={dangerFilter} onChange={e => { setDanger(e.target.value); setPage(0); }} style={selectStyle}>
          <option value="All">All Danger Levels</option>
          <option value="High">🔴 High Danger</option>
          <option value="Medium">🟠 Medium</option>
          <option value="Low">🟡 Low</option>
        </select>
        <select value={roadTypeFilter} onChange={e => { setRoadType(e.target.value); setPage(0); }} style={selectStyle}>
          {ROAD_TYPES.map(r => <option key={r} value={r}>{r === "All" ? "All Road Types" : r}</option>)}
        </select>
        <select value={seasonFilter} onChange={e => { setSeason(e.target.value); setPage(0); }} style={selectStyle}>
          {SEASONS.map(s => <option key={s} value={s}>{s === "All" ? "All Seasons" : s}</option>)}
        </select>
        <select value={timeFilter} onChange={e => { setTime(e.target.value); setPage(0); }} style={selectStyle}>
          {TIME_LABELS.map(t => <option key={t} value={t}>{t === "All" ? "All Times of Day" : t}</option>)}
        </select>
        <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={selectStyle}>
          <option value="danger_desc">Sort: Most Dangerous</option>
          <option value="danger_asc">Sort: Least Dangerous</option>
          <option value="night_desc">Sort: Highest Night Ratio</option>
          <option value="crossings">Sort: Most Crossing Events</option>
          <option value="elephants">Sort: Most Elephants</option>
        </select>
        <span style={{ marginLeft: "auto", fontSize: "13px", color: G.textMuted }}>
          {sorted.length.toLocaleString()} of {crossings.length.toLocaleString()} crossings
        </span>
      </div>

      {/* Table */}
      <div style={{ padding: "20px 32px", overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 0, background: G.card, backdropFilter: "blur(20px)", borderRadius: "16px", border: `1px solid ${G.border}`, overflow: "hidden", boxShadow: "0 8px 32px rgba(0,0,0,0.4)", tableLayout: "fixed" }}>
          <thead>
            <tr style={{ background: "rgba(239,68,68,0.10)", borderBottom: `1px solid ${G.border}` }}>
              {[
                ["Danger", "80px"], ["Corridor", "90px"], ["Road Type", "110px"],
                ["Events", "65px"], ["Elephants", "75px"], ["🌙 Night", "65px"],
                ["Peak Hours", "100px"], ["Peak Season", "120px"], ["Peak Time", "110px"],
                ["Human Dist.", "90px"], ["Traffic", "70px"],
              ].map(([h, w]) => (
                <th key={h} style={{ padding: "12px 12px", textAlign: "left", fontSize: "11px", fontWeight: "700", color: "#fca5a5", whiteSpace: "nowrap", width: w, letterSpacing: "0.5px", textTransform: "uppercase" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((c, i) => {
              const dc = dangerColor(c.danger_level);
              const db = dangerBg(c.danger_level);
              return (
                <tr key={c.crossing_id}
                  style={{ background: i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent", borderBottom: `1px solid ${G.border}`, transition: "background 0.15s" }}
                  onMouseEnter={e => e.currentTarget.style.background = "rgba(239,68,68,0.05)"}
                  onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent"}
                >
                  <td style={{ padding: "10px 12px" }}>
                    <span style={{ background: db, color: dc, padding: "3px 8px", borderRadius: "10px", fontSize: "11px", fontWeight: "700", whiteSpace: "nowrap", border: `1px solid ${dc}40`, display: "inline-block" }}>
                      {dangerIcon(c.danger_level)} {c.danger_level}<br />
                      <span style={{ fontSize: "10px", opacity: 0.8 }}>{c.danger_score.toFixed(0)}/100</span>
                    </span>
                  </td>
                  <td style={{ padding: "10px 12px", fontSize: "12px", fontWeight: "600", color: "#34d399" }}>{c.corridor_id}</td>
                  <td style={{ padding: "10px 12px", fontSize: "11px", color: G.textSecondary }}>{c.road_type}</td>
                  <td style={{ padding: "10px 12px", fontSize: "13px", textAlign: "center", fontWeight: "700", color: G.textPrimary }}>{c.crossing_count}</td>
                  <td style={{ padding: "10px 12px", fontSize: "13px", textAlign: "center", color: G.textSecondary }}>🐘×{c.elephant_count}</td>
                  <td style={{ padding: "10px 12px", textAlign: "center" }}>
                    <span style={{ fontSize: "12px", fontWeight: "700", color: c.night_ratio > 0.5 ? "#ef4444" : "#10b981" }}>
                      {(c.night_ratio * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td style={{ padding: "10px 12px", fontSize: "11px", color: G.textMuted }}>{c.peak_hours.map(h => `${h}:00`).join(", ")}</td>
                  <td style={{ padding: "10px 12px", fontSize: "11px", color: G.textMuted }}>{c.peak_season}</td>
                  <td style={{ padding: "10px 12px", fontSize: "11px", color: G.textMuted }}>{c.peak_time_of_day}</td>
                  <td style={{ padding: "10px 12px", fontSize: "12px", textAlign: "center", color: G.textSecondary }}>{(c.avg_human_distance / 1000).toFixed(1)} km</td>
                  <td style={{ padding: "10px 12px", textAlign: "center" }}>
                    <span style={{ fontSize: "11px", fontWeight: "700", color: c.traffic_exposure > 0.6 ? "#ef4444" : c.traffic_exposure > 0.3 ? "#f59e0b" : "#10b981" }}>
                      {(c.traffic_exposure * 100).toFixed(0)}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "6px", marginTop: "24px", flexWrap: "wrap" }}>
            {[
              { label: "« First", disabled: page === 0,              action: () => setPage(0) },
              { label: "← Prev",  disabled: page === 0,              action: () => setPage(p => p - 1) },
            ].map(btn => (
              <button key={btn.label} disabled={btn.disabled} onClick={btn.action} style={{ padding: "7px 14px", borderRadius: "10px", border: `1px solid ${G.border}`, cursor: btn.disabled ? "not-allowed" : "pointer", background: btn.disabled ? "rgba(255,255,255,0.03)" : G.inputBg, color: btn.disabled ? G.textMuted : G.textPrimary, fontSize: "12px", fontWeight: "500", backdropFilter: "blur(10px)" }}>
                {btn.label}
              </button>
            ))}
            {[...Array(Math.min(7, totalPages))].map((_, j) => {
              const pg = Math.max(0, Math.min(page - 3, totalPages - 7)) + j;
              return (
                <button key={pg} onClick={() => setPage(pg)} style={{ padding: "7px 12px", borderRadius: "10px", border: `1px solid ${pg === page ? "#ef444460" : G.border}`, cursor: "pointer", background: pg === page ? "rgba(239,68,68,0.2)" : G.inputBg, color: pg === page ? "#f87171" : G.textSecondary, fontWeight: pg === page ? "700" : "500", fontSize: "12px", backdropFilter: "blur(10px)" }}>
                  {pg + 1}
                </button>
              );
            })}
            {[
              { label: "Next →", disabled: page >= totalPages - 1, action: () => setPage(p => p + 1) },
              { label: "Last »", disabled: page >= totalPages - 1, action: () => setPage(totalPages - 1) },
            ].map(btn => (
              <button key={btn.label} disabled={btn.disabled} onClick={btn.action} style={{ padding: "7px 14px", borderRadius: "10px", border: `1px solid ${G.border}`, cursor: btn.disabled ? "not-allowed" : "pointer", background: btn.disabled ? "rgba(255,255,255,0.03)" : G.inputBg, color: btn.disabled ? G.textMuted : G.textPrimary, fontSize: "12px", fontWeight: "500", backdropFilter: "blur(10px)" }}>
                {btn.label}
              </button>
            ))}
            <span style={{ fontSize: "13px", color: G.textMuted }}>Page {page + 1} of {totalPages} ({sorted.length.toLocaleString()} results)</span>
          </div>
        )}
        {paged.length === 0 && (
          <div style={{ textAlign: "center", padding: "48px", color: G.textMuted, fontSize: "15px" }}>No road crossings match the current filters.</div>
        )}
      </div>
    </div>
  );
}
