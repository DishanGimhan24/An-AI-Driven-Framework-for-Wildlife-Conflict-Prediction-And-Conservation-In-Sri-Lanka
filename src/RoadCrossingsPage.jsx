import { useState, useEffect, useMemo } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { DISHAN_API } from "./apiConfig";

const PAGE_SIZE = 25;

const SEASONS = ["All", "Dry", "Inter-Monsoon", "Southwest-Monsoon", "Inter-Monsoon-2"];
const ROAD_TYPES = ["All", "Major Road", "Secondary Road", "Minor Road"];
const TIME_LABELS = ["All", "Night", "Early Morning", "Midday", "Afternoon", "Evening"];

const dangerColor  = (lvl) => lvl === "High" ? "#d32f2f" : lvl === "Medium" ? "#f57c00" : "#fbc02d";
const dangerBg     = (lvl) => lvl === "High" ? "#ffebee" : lvl === "Medium" ? "#fff3e0" : "#fffde7";
const dangerIcon   = (lvl) => lvl === "High" ? "🔴" : lvl === "Medium" ? "🟠" : "🟡";

const mkIcon = (lvl) => L.divIcon({
  html: `<div style="font-size:16px;text-align:center;line-height:1;filter:drop-shadow(0 1px 3px rgba(0,0,0,0.5))">${dangerIcon(lvl)}</div>`,
  className: "",
  iconSize: [20, 20],
  iconAnchor: [10, 10],
  popupAnchor: [0, -12]
});

export default function RoadCrossingsPage() {
  const [crossings, setCrossings] = useState([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [summary, setSummary]     = useState(null);
  const [search, setSearch]       = useState("");
  const [dangerFilter, setDanger] = useState("All");
  const [roadTypeFilter, setRoadType] = useState("All");
  const [seasonFilter, setSeason] = useState("All");
  const [timeFilter, setTime]     = useState("All");
  const [sortBy, setSortBy]       = useState("danger_desc");
  const [page, setPage]           = useState(0);
  const [mapVisible, setMapVisible] = useState(true);

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

  // Filter
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

  // Sort
  const sorted = useMemo(() => [...filtered].sort((a, b) => {
    if (sortBy === "danger_desc")   return b.danger_score - a.danger_score;
    if (sortBy === "danger_asc")    return a.danger_score - b.danger_score;
    if (sortBy === "night_desc")    return b.night_ratio - a.night_ratio;
    if (sortBy === "crossings")     return b.crossing_count - a.crossing_count;
    if (sortBy === "elephants")     return b.elephant_count - a.elephant_count;
    return 0;
  }), [filtered, sortBy]);

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
  const paged = sorted.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

  // Map markers: only show high+medium (too many low ones to render)
  const mapMarkers = useMemo(() =>
    sorted.filter(c => c.danger_level !== "Low").slice(0, 400),
    [sorted]
  );

  const inputStyle = { padding: "8px 12px", fontSize: "13px", borderRadius: "6px", border: "1px solid #ccc", cursor: "pointer", outline: "none", backgroundColor: "white" };

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", fontSize: "16px", color: "#666" }}>
      <span style={{ marginRight: "10px", fontSize: "30px" }}>🚗</span> Loading road crossings...
    </div>
  );
  if (error) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", color: "#d32f2f" }}>
      ⚠️ {error}. Make sure the API server is running.
    </div>
  );

  const dnCounts = summary?.by_danger_level ?? {};
  const rtCounts = summary?.by_road_type ?? {};

  return (
    <div style={{ minHeight: "calc(100vh - 60px)", backgroundColor: "#f4f6fb" }}>

      {/* Page header */}
      <div style={{ background: "linear-gradient(90deg, #7f0000, #B71C1C)", color: "white", padding: "24px 32px" }}>
        <h1 style={{ margin: "0 0 6px 0", fontSize: "22px", fontWeight: "700" }}>🚗 Road Crossing Analysis</h1>
        <p style={{ margin: 0, fontSize: "13px", opacity: 0.85 }}>
          {crossings.length.toLocaleString()} crossings where elephant corridors intersect roads · Showing {sorted.length} filtered
        </p>
      </div>

      {/* Summary row */}
      <div style={{ backgroundColor: "white", borderBottom: "1px solid #e0e0e0", padding: "12px 32px", display: "flex", gap: "24px", flexWrap: "wrap", alignItems: "center" }}>
        <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
          {[
            { label: "🔴 High",   count: dnCounts.High   ?? 0, color: "#d32f2f" },
            { label: "🟠 Medium", count: dnCounts.Medium ?? 0, color: "#f57c00" },
            { label: "🟡 Low",    count: dnCounts.Low    ?? 0, color: "#f9a825" },
          ].map(d => (
            <div key={d.label} style={{ textAlign: "center" }}>
              <div style={{ fontSize: "20px", fontWeight: "700", color: d.color }}>{d.count.toLocaleString()}</div>
              <div style={{ fontSize: "11px", color: "#666" }}>{d.label} Danger</div>
            </div>
          ))}
        </div>
        <div style={{ width: "1px", backgroundColor: "#e0e0e0", height: "36px" }} />
        <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
          {Object.entries(rtCounts).map(([type, cnt]) => (
            <div key={type} style={{ textAlign: "center" }}>
              <div style={{ fontSize: "18px", fontWeight: "700", color: "#555" }}>{cnt.toLocaleString()}</div>
              <div style={{ fontSize: "11px", color: "#666" }}>{type}</div>
            </div>
          ))}
        </div>
        <div style={{ marginLeft: "auto" }}>
          <button onClick={() => setMapVisible(v => !v)} style={{ padding: "7px 14px", borderRadius: "6px", border: "1px solid #B71C1C", backgroundColor: mapVisible ? "#B71C1C" : "white", color: mapVisible ? "white" : "#B71C1C", fontSize: "12px", fontWeight: "600", cursor: "pointer" }}>
            {mapVisible ? "🗺️ Hide Map" : "🗺️ Show Map"}
          </button>
        </div>
      </div>

      {/* Mini map */}
      {mapVisible && (
        <div style={{ height: "300px", borderBottom: "1px solid #e0e0e0" }}>
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
      <div style={{ backgroundColor: "white", borderBottom: "1px solid #e0e0e0", padding: "12px 32px", display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center" }}>
        <input
          placeholder="🔍 Search crossing or corridor ID..."
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(0); }}
          style={{ ...inputStyle, minWidth: "220px" }}
        />
        <select value={dangerFilter} onChange={e => { setDanger(e.target.value); setPage(0); }} style={inputStyle}>
          <option value="All">All Danger Levels</option>
          <option value="High">🔴 High Danger</option>
          <option value="Medium">🟠 Medium</option>
          <option value="Low">🟡 Low</option>
        </select>
        <select value={roadTypeFilter} onChange={e => { setRoadType(e.target.value); setPage(0); }} style={inputStyle}>
          {ROAD_TYPES.map(r => <option key={r} value={r}>{r === "All" ? "All Road Types" : r}</option>)}
        </select>
        <select value={seasonFilter} onChange={e => { setSeason(e.target.value); setPage(0); }} style={inputStyle}>
          {SEASONS.map(s => <option key={s} value={s}>{s === "All" ? "All Seasons" : s}</option>)}
        </select>
        <select value={timeFilter} onChange={e => { setTime(e.target.value); setPage(0); }} style={inputStyle}>
          {TIME_LABELS.map(t => <option key={t} value={t}>{t === "All" ? "All Times of Day" : t}</option>)}
        </select>
        <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={inputStyle}>
          <option value="danger_desc">Sort: Most Dangerous</option>
          <option value="danger_asc">Sort: Least Dangerous</option>
          <option value="night_desc">Sort: Highest Night Ratio</option>
          <option value="crossings">Sort: Most Crossing Events</option>
          <option value="elephants">Sort: Most Elephants</option>
        </select>
        <span style={{ marginLeft: "auto", fontSize: "13px", color: "#888" }}>
          {sorted.length.toLocaleString()} of {crossings.length.toLocaleString()} crossings
        </span>
      </div>

      {/* Table */}
      <div style={{ padding: "20px 32px", overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", backgroundColor: "white", borderRadius: "12px", boxShadow: "0 2px 10px rgba(0,0,0,0.07)", overflow: "hidden", tableLayout: "fixed" }}>
          <thead>
            <tr style={{ background: "linear-gradient(90deg, #7f0000, #B71C1C)", color: "white" }}>
              {[
                ["Danger", "80px"],
                ["Corridor", "90px"],
                ["Road Type", "110px"],
                ["Events", "65px"],
                ["Elephants", "75px"],
                ["🌙 Night", "65px"],
                ["Peak Hours", "100px"],
                ["Peak Season", "120px"],
                ["Peak Time", "110px"],
                ["Human Dist.", "90px"],
                ["Traffic", "70px"],
              ].map(([h, w]) => (
                <th key={h} style={{ padding: "11px 12px", textAlign: "left", fontSize: "11px", fontWeight: "600", whiteSpace: "nowrap", width: w }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((c, i) => {
              const dc = dangerColor(c.danger_level);
              const db = dangerBg(c.danger_level);
              return (
                <tr key={c.crossing_id}
                  style={{ backgroundColor: i % 2 === 0 ? "#fafafa" : "white", borderBottom: "1px solid #f0f0f0" }}
                  onMouseEnter={e => e.currentTarget.style.backgroundColor = "#fff3f3"}
                  onMouseLeave={e => e.currentTarget.style.backgroundColor = i % 2 === 0 ? "#fafafa" : "white"}
                >
                  <td style={{ padding: "10px 12px" }}>
                    <span style={{ backgroundColor: db, color: dc, padding: "3px 8px", borderRadius: "10px", fontSize: "11px", fontWeight: "700", whiteSpace: "nowrap" }}>
                      {dangerIcon(c.danger_level)} {c.danger_level}<br />
                      <span style={{ fontSize: "10px", opacity: 0.8 }}>{c.danger_score.toFixed(0)}/100</span>
                    </span>
                  </td>
                  <td style={{ padding: "10px 12px", fontSize: "12px", fontWeight: "600", color: "#4A148C" }}>{c.corridor_id}</td>
                  <td style={{ padding: "10px 12px", fontSize: "11px", color: "#555" }}>{c.road_type}</td>
                  <td style={{ padding: "10px 12px", fontSize: "13px", textAlign: "center", fontWeight: "600" }}>{c.crossing_count}</td>
                  <td style={{ padding: "10px 12px", fontSize: "13px", textAlign: "center" }}>🐘×{c.elephant_count}</td>
                  <td style={{ padding: "10px 12px", textAlign: "center" }}>
                    <span style={{ fontSize: "12px", fontWeight: "700", color: c.night_ratio > 0.5 ? "#d32f2f" : "#388e3c" }}>
                      {(c.night_ratio * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td style={{ padding: "10px 12px", fontSize: "11px", color: "#555" }}>{c.peak_hours.map(h => `${h}:00`).join(", ")}</td>
                  <td style={{ padding: "10px 12px", fontSize: "11px", color: "#555" }}>{c.peak_season}</td>
                  <td style={{ padding: "10px 12px", fontSize: "11px", color: "#555" }}>{c.peak_time_of_day}</td>
                  <td style={{ padding: "10px 12px", fontSize: "12px", textAlign: "center" }}>{(c.avg_human_distance / 1000).toFixed(1)} km</td>
                  <td style={{ padding: "10px 12px", textAlign: "center" }}>
                    <span style={{ fontSize: "11px", color: c.traffic_exposure > 0.6 ? "#d32f2f" : c.traffic_exposure > 0.3 ? "#f57c00" : "#388e3c", fontWeight: "700" }}>
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
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "8px", marginTop: "20px", flexWrap: "wrap" }}>
            <button disabled={page === 0} onClick={() => setPage(0)} style={{ padding: "6px 12px", borderRadius: "6px", border: "1px solid #ccc", cursor: page === 0 ? "not-allowed" : "pointer", backgroundColor: "white", fontSize: "12px" }}>« First</button>
            <button disabled={page === 0} onClick={() => setPage(p => p - 1)} style={{ padding: "6px 12px", borderRadius: "6px", border: "1px solid #ccc", cursor: page === 0 ? "not-allowed" : "pointer", backgroundColor: "white", fontSize: "12px" }}>← Prev</button>
            {[...Array(Math.min(7, totalPages))].map((_, j) => {
              const pg = Math.max(0, Math.min(page - 3, totalPages - 7)) + j;
              return (
                <button key={pg} onClick={() => setPage(pg)} style={{ padding: "6px 12px", borderRadius: "6px", border: "1px solid #ccc", cursor: "pointer", backgroundColor: pg === page ? "#B71C1C" : "white", color: pg === page ? "white" : "#333", fontWeight: pg === page ? "700" : "400", fontSize: "12px" }}>
                  {pg + 1}
                </button>
              );
            })}
            <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)} style={{ padding: "6px 12px", borderRadius: "6px", border: "1px solid #ccc", cursor: page >= totalPages - 1 ? "not-allowed" : "pointer", backgroundColor: "white", fontSize: "12px" }}>Next →</button>
            <button disabled={page >= totalPages - 1} onClick={() => setPage(totalPages - 1)} style={{ padding: "6px 12px", borderRadius: "6px", border: "1px solid #ccc", cursor: page >= totalPages - 1 ? "not-allowed" : "pointer", backgroundColor: "white", fontSize: "12px" }}>Last »</button>
            <span style={{ fontSize: "13px", color: "#888" }}>Page {page + 1} of {totalPages} ({sorted.length.toLocaleString()} results)</span>
          </div>
        )}
        {paged.length === 0 && (
          <div style={{ textAlign: "center", padding: "40px", color: "#888", fontSize: "15px" }}>No road crossings match the current filters.</div>
        )}
      </div>
    </div>
  );
}
