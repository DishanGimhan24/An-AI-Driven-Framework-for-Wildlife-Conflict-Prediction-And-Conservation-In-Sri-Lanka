import { useState, useEffect } from "react";
import axios from "axios";

const PAGE_SIZE = 20;

const safetyLabel = (s) => s < 30 ? "High" : s < 60 ? "Medium" : s < 80 ? "Moderate" : "Low";
const safetyColor = (s) => s < 30 ? "#d32f2f" : s < 60 ? "#f57c00" : s < 80 ? "#fbc02d" : "#388e3c";
const safetyBg    = (s) => s < 30 ? "#ffebee" : s < 60 ? "#fff3e0" : s < 80 ? "#fffde7" : "#e8f5e9";

export default function CorridorsPage() {
  const [corridors, setCorridors] = useState([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [search, setSearch]       = useState("");
  const [dangerFilter, setDanger] = useState("All");
  const [sortBy, setSortBy]       = useState("safety_asc");
  const [page, setPage]           = useState(0);
  const [expanded, setExpanded]   = useState(null);

  useEffect(() => {
    axios.get("http://localhost:8000/corridors")
      .then(r => { setCorridors(r.data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const filtered = corridors.filter(c => {
    if (search && !c.corridor_id.includes(search)) return false;
    if (dangerFilter !== "All" && safetyLabel(c.safety_score) !== dangerFilter) return false;
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === "safety_asc")    return a.safety_score - b.safety_score;
    if (sortBy === "safety_desc")   return b.safety_score - a.safety_score;
    if (sortBy === "usage")         return b.usage_count - a.usage_count;
    if (sortBy === "crossings")     return b.crossing_count - a.crossing_count;
    if (sortBy === "distance")      return b.distance_meters - a.distance_meters;
    return 0;
  });

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
  const paged = sorted.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

  const inputStyle = { padding: "8px 12px", fontSize: "13px", borderRadius: "6px", border: "1px solid #ccc", cursor: "pointer", outline: "none" };

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", fontSize: "16px", color: "#666" }}>
      <span style={{ marginRight: "10px", fontSize: "30px" }}>🛤️</span> Loading corridors...
    </div>
  );
  if (error) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", color: "#d32f2f" }}>
      ⚠️ {error}. Make sure the API server is running.
    </div>
  );

  return (
    <div style={{ minHeight: "calc(100vh - 60px)", backgroundColor: "#f4f6fb" }}>

      {/* Page header */}
      <div style={{ backgroundColor: "#4A148C", color: "white", padding: "24px 32px" }}>
        <h1 style={{ margin: "0 0 6px 0", fontSize: "22px", fontWeight: "700" }}>🛤️ Elephant Movement Corridors</h1>
        <p style={{ margin: 0, fontSize: "13px", opacity: 0.85 }}>
          {corridors.length} movement paths between hotspot nodes · Showing {sorted.length} filtered results
        </p>
      </div>

      {/* Summary stats */}
      <div style={{ backgroundColor: "white", borderBottom: "1px solid #e0e0e0", padding: "10px 32px", display: "flex", gap: "32px", flexWrap: "wrap" }}>
        {[
          { label: "Total Corridors", value: corridors.length, color: "#4A148C" },
          { label: "High Danger", value: corridors.filter(c => safetyLabel(c.safety_score) === "High").length, color: "#d32f2f" },
          { label: "Medium Danger", value: corridors.filter(c => safetyLabel(c.safety_score) === "Medium").length, color: "#f57c00" },
          { label: "Avg Safety Score", value: corridors.length ? (corridors.reduce((a, c) => a + c.safety_score, 0) / corridors.length).toFixed(1) : 0, color: "#388e3c" },
        ].map(s => (
          <div key={s.label} style={{ textAlign: "center", padding: "6px 0" }}>
            <div style={{ fontSize: "20px", fontWeight: "700", color: s.color }}>{s.value}</div>
            <div style={{ fontSize: "11px", color: "#666" }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Filter bar */}
      <div style={{ backgroundColor: "white", borderBottom: "1px solid #e0e0e0", padding: "12px 32px", display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center" }}>
        <input
          placeholder="🔍 Search corridor ID (e.g. 33-34)..."
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(0); }}
          style={{ ...inputStyle, minWidth: "220px" }}
        />
        <select value={dangerFilter} onChange={e => { setDanger(e.target.value); setPage(0); }} style={inputStyle}>
          <option value="All">All Danger Levels</option>
          <option value="High">🔴 High Danger</option>
          <option value="Medium">🟠 Medium</option>
          <option value="Moderate">🟡 Moderate</option>
          <option value="Low">🟢 Low</option>
        </select>
        <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={inputStyle}>
          <option value="safety_asc">Sort: Most Dangerous First</option>
          <option value="safety_desc">Sort: Safest First</option>
          <option value="usage">Sort: Most Used</option>
          <option value="crossings">Sort: Most Crossings</option>
          <option value="distance">Sort: Longest First</option>
        </select>
        <span style={{ marginLeft: "auto", fontSize: "13px", color: "#888" }}>
          {sorted.length} of {corridors.length} corridors
        </span>
      </div>

      {/* Table */}
      <div style={{ padding: "20px 32px", overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", backgroundColor: "white", borderRadius: "12px", boxShadow: "0 2px 10px rgba(0,0,0,0.07)", overflow: "hidden" }}>
          <thead>
            <tr style={{ backgroundColor: "#4A148C", color: "white" }}>
              {["Corridor ID", "Route", "Length", "Usage", "Crossings", "Active Hours", "Human Dist.", "Road Crossings", "Danger"].map(h => (
                <th key={h} style={{ padding: "12px 14px", textAlign: "left", fontSize: "12px", fontWeight: "600", whiteSpace: "nowrap" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((c, i) => {
              const isExp = expanded === c.corridor_id;
              const sc = safetyColor(c.safety_score);
              const sb = safetyBg(c.safety_score);
              const roadCrossings = c.road_crossings ?? [];
              const rcHigh = roadCrossings.filter(r => r.danger_level === "High").length;
              return (
                <>
                  <tr key={c.corridor_id}
                    onClick={() => setExpanded(isExp ? null : c.corridor_id)}
                    style={{ backgroundColor: i % 2 === 0 ? "#fafafa" : "white", cursor: "pointer", borderBottom: "1px solid #f0f0f0" }}
                    onMouseEnter={e => e.currentTarget.style.backgroundColor = "#f3e5f5"}
                    onMouseLeave={e => e.currentTarget.style.backgroundColor = i % 2 === 0 ? "#fafafa" : "white"}
                  >
                    <td style={{ padding: "11px 14px", fontSize: "13px", fontWeight: "700", color: "#4A148C" }}>{c.corridor_id}</td>
                    <td style={{ padding: "11px 14px", fontSize: "12px", color: "#555", whiteSpace: "nowrap" }}>
                      Node <strong>{c.from_node}</strong> → <strong>{c.to_node}</strong>
                    </td>
                    <td style={{ padding: "11px 14px", fontSize: "13px", textAlign: "center" }}>{(c.distance_meters / 1000).toFixed(1)} km</td>
                    <td style={{ padding: "11px 14px", fontSize: "13px", textAlign: "center", fontWeight: "600" }}>×{c.usage_count}</td>
                    <td style={{ padding: "11px 14px", fontSize: "13px", textAlign: "center" }}>{c.crossing_count}</td>
                    <td style={{ padding: "11px 14px", fontSize: "11px", color: "#555" }}>{c.active_hours.slice(0, 4).map(h => `${h}:00`).join(", ")}{c.active_hours.length > 4 ? " ..." : ""}</td>
                    <td style={{ padding: "11px 14px", fontSize: "13px", textAlign: "center" }}>{(c.avg_human_distance / 1000).toFixed(1)} km</td>
                    <td style={{ padding: "11px 14px", textAlign: "center" }}>
                      <span style={{ fontSize: "12px" }}>
                        {roadCrossings.length > 0 ? (
                          <span>
                            {roadCrossings.length}
                            {rcHigh > 0 && <span style={{ marginLeft: "5px", backgroundColor: "#ffebee", color: "#d32f2f", padding: "1px 6px", borderRadius: "8px", fontSize: "10px", fontWeight: "700" }}>🔴 {rcHigh} high</span>}
                          </span>
                        ) : <span style={{ color: "#aaa" }}>—</span>}
                      </span>
                    </td>
                    <td style={{ padding: "11px 14px" }}>
                      <span style={{ backgroundColor: sb, color: sc, padding: "3px 10px", borderRadius: "10px", fontSize: "11px", fontWeight: "700", whiteSpace: "nowrap" }}>
                        {safetyLabel(c.safety_score)} · {c.safety_score.toFixed(0)}
                      </span>
                    </td>
                  </tr>

                  {/* Expanded row: road crossings detail */}
                  {isExp && (
                    <tr key={`exp-${c.corridor_id}`} style={{ backgroundColor: "#f9f0ff" }}>
                      <td colSpan={9} style={{ padding: "14px 24px" }}>
                        <div style={{ display: "flex", gap: "20px", flexWrap: "wrap", fontSize: "12px", color: "#444", marginBottom: "10px" }}>
                          <div><strong>Bidirectional:</strong> {c.bidirectional ? "Yes" : "No"}</div>
                          <div><strong>Safety Score:</strong> {c.safety_score.toFixed(1)} / 100</div>
                          <div><strong>Elephants Used:</strong> {c.elephants ? c.elephants.join(", ") : "—"}</div>
                        </div>
                        {roadCrossings.length > 0 ? (
                          <>
                            <div style={{ fontWeight: "700", marginBottom: "8px", fontSize: "12px", color: "#4A148C" }}>🚗 Road Crossings along this corridor:</div>
                            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                              {roadCrossings.slice(0, 8).map((rc, j) => {
                                const dc = rc.danger_level === "High" ? "#d32f2f" : rc.danger_level === "Medium" ? "#f57c00" : "#fbc02d";
                                return (
                                  <div key={j} style={{ backgroundColor: "white", border: `1px solid ${dc}`, borderRadius: "8px", padding: "8px 12px", fontSize: "11px", minWidth: "160px" }}>
                                    <div style={{ fontWeight: "700", color: dc, marginBottom: "4px" }}>{rc.danger_level} · Score {rc.danger_score.toFixed(0)}</div>
                                    <div style={{ color: "#555" }}>{rc.road_type}</div>
                                    <div style={{ color: "#555" }}>🌙 Night: {(rc.night_ratio * 100).toFixed(0)}%</div>
                                    <div style={{ color: "#555" }}>🌿 Season: {rc.peak_season}</div>
                                    <div style={{ color: "#555" }}>⏰ Peak: {rc.peak_hours.map(h => `${h}:00`).join(", ")}</div>
                                  </div>
                                );
                              })}
                              {roadCrossings.length > 8 && (
                                <div style={{ display: "flex", alignItems: "center", fontSize: "11px", color: "#888" }}>+{roadCrossings.length - 8} more...</div>
                              )}
                            </div>
                          </>
                        ) : (
                          <div style={{ fontSize: "12px", color: "#888" }}>No road crossings detected along this corridor.</div>
                        )}
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
          </tbody>
        </table>

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "8px", marginTop: "20px" }}>
            <button disabled={page === 0} onClick={() => setPage(p => p - 1)} style={{ padding: "7px 16px", borderRadius: "6px", border: "1px solid #ccc", cursor: page === 0 ? "not-allowed" : "pointer", backgroundColor: page === 0 ? "#f5f5f5" : "white", fontSize: "13px" }}>← Prev</button>
            <span style={{ fontSize: "13px", color: "#666" }}>Page {page + 1} of {totalPages}</span>
            <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)} style={{ padding: "7px 16px", borderRadius: "6px", border: "1px solid #ccc", cursor: page >= totalPages - 1 ? "not-allowed" : "pointer", backgroundColor: page >= totalPages - 1 ? "#f5f5f5" : "white", fontSize: "13px" }}>Next →</button>
          </div>
        )}
        {paged.length === 0 && (
          <div style={{ textAlign: "center", padding: "40px", color: "#888", fontSize: "15px" }}>No corridors match the current filters.</div>
        )}
      </div>
    </div>
  );
}
