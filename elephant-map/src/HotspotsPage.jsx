import { useState, useEffect } from "react";
import axios from "axios";
import { DISHAN_API } from "./apiConfig";

const PAGE_SIZE = 20;

const DISTRICTS = [
  "All", "Galle", "Matara", "Hambantota", "Monaragala", "Badulla",
  "Ratnapura", "Kegalle", "Kandy", "Matale", "Nuwara Eliya",
  "Anuradhapura", "Polonnaruwa", "Ampara", "Batticaloa",
  "Trincomalee", "Colombo", "Kalutara", "Gampaha",
  "Kurunegala", "Puttalam", "Other"
];

const getDistrict = (lat, lon) => {
  if (lat >= 6.0 && lat <= 6.5 && lon >= 80.0 && lon <= 81.5) {
    if (lon < 80.5) return "Galle";
    if (lon < 81.0) return "Matara";
    return "Hambantota";
  }
  if (lat >= 6.5 && lat <= 7.5 && lon >= 81.0 && lon <= 81.8) return lat < 6.8 ? "Monaragala" : "Badulla";
  if (lat >= 6.5 && lat <= 7.5 && lon >= 80.2 && lon <= 81.0) return lat < 7.0 ? "Ratnapura" : "Kegalle";
  if (lat >= 7.0 && lat <= 7.5 && lon >= 80.5 && lon <= 81.3) { if (lon < 80.8) return "Kandy"; if (lat < 7.3) return "Matale"; return "Nuwara Eliya"; }
  if (lat >= 7.5 && lat <= 8.5 && lon >= 80.0 && lon <= 81.3) return lon < 80.5 ? "Anuradhapura" : "Polonnaruwa";
  if (lat >= 7.0 && lon >= 81.0) { if (lat < 7.5) return "Ampara"; if (lat < 8.5) return "Batticaloa"; return "Trincomalee"; }
  if (lat >= 6.5 && lat <= 7.5 && lon >= 79.5 && lon <= 80.5) { if (lat > 7.0 && lon > 79.8 && lon < 80.2) return "Colombo"; if (lat < 7.0) return "Kalutara"; return "Gampaha"; }
  if (lat >= 7.0 && lat <= 8.5 && lon >= 79.5 && lon <= 80.5) return lat < 7.8 ? "Kurunegala" : "Puttalam";
  return "Other";
};

const safetyLabel = (s) => s < 30 ? "High" : s < 60 ? "Medium" : s < 80 ? "Moderate" : "Low";
const safetyColor = (s) => s < 30 ? "#d32f2f" : s < 60 ? "#f57c00" : s < 80 ? "#fbc02d" : "#388e3c";
const safetyBg    = (s) => s < 30 ? "#ffebee" : s < 60 ? "#fff3e0" : s < 80 ? "#fffde7" : "#e8f5e9";

export default function HotspotsPage() {
  const [nodes, setNodes]               = useState([]);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);
  const [search, setSearch]             = useState("");
  const [districtFilter, setDistrict]   = useState("All");
  const [dangerFilter, setDanger]       = useState("All");
  const [protectedFilter, setProtected] = useState("All");
  const [eleFilter, setEle]             = useState("All");
  const [sortBy, setSortBy]             = useState("safety_asc");
  const [page, setPage]                 = useState(0);
  const [expanded, setExpanded]         = useState(null);

  useEffect(() => {
    axios.get(`${DISHAN_API}/nodes`)
      .then(r => {
        const enriched = r.data.map(n => ({ ...n, district: getDistrict(n.center_lat, n.center_lon) }));
        setNodes(enriched);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  // Filter
  const filtered = nodes.filter(n => {
    if (search && !String(n.node_id).includes(search) && !n.district.toLowerCase().includes(search.toLowerCase())) return false;
    if (districtFilter !== "All" && n.district !== districtFilter) return false;
    if (dangerFilter !== "All" && safetyLabel(n.safety_score) !== dangerFilter) return false;
    if (protectedFilter !== "All" && String(n.protected) !== protectedFilter) return false;
    if (eleFilter === "1" && n.elephant_count !== 1) return false;
    if (eleFilter === "2" && n.elephant_count !== 2) return false;
    if (eleFilter === "3+" && n.elephant_count < 3) return false;
    return true;
  });

  // Sort
  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === "safety_asc")    return a.safety_score - b.safety_score;
    if (sortBy === "safety_desc")   return b.safety_score - a.safety_score;
    if (sortBy === "sightings")     return b.sighting_count - a.sighting_count;
    if (sortBy === "elephants")     return b.elephant_count - a.elephant_count;
    if (sortBy === "human_dist")    return a.avg_human_distance - b.avg_human_distance;
    return a.node_id - b.node_id;
  });

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
  const paged = sorted.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

  const inputStyle = { padding: "8px 12px", fontSize: "13px", borderRadius: "6px", border: "1px solid #ccc", cursor: "pointer", outline: "none" };

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", fontSize: "16px", color: "#666" }}>
      <span style={{ marginRight: "10px", fontSize: "30px" }}>🐘</span> Loading hotspots...
    </div>
  );

  if (error) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", color: "#d32f2f" }}>
      ⚠️ {error}. Make sure API server is running.
    </div>
  );

  return (
    <div style={{ minHeight: "calc(100vh - 60px)", backgroundColor: "#f4f6fb" }}>

      {/* Page header */}
      <div style={{ backgroundColor: "#1B5E20", color: "white", padding: "24px 32px" }}>
        <h1 style={{ margin: "0 0 6px 0", fontSize: "22px", fontWeight: "700" }}>🐘 Elephant Hotspot Zones</h1>
        <p style={{ margin: 0, fontSize: "13px", opacity: 0.85 }}>
          {nodes.length} high-use elephant zones detected via DBSCAN clustering · Showing {sorted.length} filtered results
        </p>
      </div>

      {/* Filter bar */}
      <div style={{ backgroundColor: "white", borderBottom: "1px solid #e0e0e0", padding: "14px 32px", display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center" }}>
        <input
          placeholder="🔍 Search node ID or district..."
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(0); }}
          style={{ ...inputStyle, minWidth: "220px" }}
        />
        <select value={districtFilter} onChange={e => { setDistrict(e.target.value); setPage(0); }} style={inputStyle}>
          {DISTRICTS.map(d => <option key={d} value={d}>{d === "All" ? "All Districts" : d}</option>)}
        </select>
        <select value={dangerFilter} onChange={e => { setDanger(e.target.value); setPage(0); }} style={inputStyle}>
          <option value="All">All Danger Levels</option>
          <option value="High">🔴 High Danger</option>
          <option value="Medium">🟠 Medium</option>
          <option value="Moderate">🟡 Moderate</option>
          <option value="Low">🟢 Low</option>
        </select>
        <select value={eleFilter} onChange={e => { setEle(e.target.value); setPage(0); }} style={inputStyle}>
          <option value="All">All Elephant Counts</option>
          <option value="1">1 Elephant</option>
          <option value="2">2 Elephants</option>
          <option value="3+">3+ Elephants</option>
        </select>
        <select value={protectedFilter} onChange={e => { setProtected(e.target.value); setPage(0); }} style={inputStyle}>
          <option value="All">All Areas</option>
          <option value="1">Protected Only</option>
          <option value="0">Unprotected Only</option>
        </select>
        <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={inputStyle}>
          <option value="safety_asc">Sort: Most Dangerous First</option>
          <option value="safety_desc">Sort: Safest First</option>
          <option value="sightings">Sort: Most Sightings</option>
          <option value="elephants">Sort: Most Elephants</option>
          <option value="human_dist">Sort: Closest to Humans</option>
          <option value="id">Sort: Node ID</option>
        </select>
        <span style={{ marginLeft: "auto", fontSize: "13px", color: "#888" }}>
          {sorted.length} out of {nodes.length} zones
        </span>
      </div>

      {/* Table */}
      <div style={{ padding: "20px 32px", overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", backgroundColor: "white", borderRadius: "12px", boxShadow: "0 2px 10px rgba(0,0,0,0.07)", overflow: "hidden" }}>
          <thead>
            <tr style={{ backgroundColor: "#1B5E20", color: "white" }}>
              {["Node ID", "District", "Coordinates", "Elephants", "Sightings", "Active Hours", "NDVI", "Human Dist.", "Protected", "Danger"].map(h => (
                <th key={h} style={{ padding: "12px 14px", textAlign: "left", fontSize: "12px", fontWeight: "600", whiteSpace: "nowrap" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((n, i) => {
              const isExp = expanded === n.node_id;
              const sc = safetyColor(n.safety_score);
              const sb = safetyBg(n.safety_score);
              return (
                <>
                  <tr key={n.node_id}
                    onClick={() => setExpanded(isExp ? null : n.node_id)}
                    style={{ backgroundColor: i % 2 === 0 ? "#fafafa" : "white", cursor: "pointer", borderBottom: "1px solid #f0f0f0", transition: "background 0.15s" }}
                    onMouseEnter={e => e.currentTarget.style.backgroundColor = "#e8f5e9"}
                    onMouseLeave={e => e.currentTarget.style.backgroundColor = i % 2 === 0 ? "#fafafa" : "white"}
                  >
                    <td style={{ padding: "11px 14px", fontSize: "13px", fontWeight: "700", color: "#1B5E20" }}>#{n.node_id}</td>
                    <td style={{ padding: "11px 14px", fontSize: "13px" }}>{n.district}</td>
                    <td style={{ padding: "11px 14px", fontSize: "11px", color: "#666", fontFamily: "monospace" }}>
                      {n.center_lat.toFixed(4)}, {n.center_lon.toFixed(4)}
                    </td>
                    <td style={{ padding: "11px 14px", fontSize: "13px", textAlign: "center" }}>
                      {"🐘".repeat(Math.min(n.elephant_count, 5))} <span style={{ fontSize: "11px", color: "#888" }}>×{n.elephant_count}</span>
                    </td>
                    <td style={{ padding: "11px 14px", fontSize: "13px", textAlign: "center", fontWeight: "600" }}>{n.sighting_count}</td>
                    <td style={{ padding: "11px 14px", fontSize: "11px", color: "#555" }}>{n.active_hours.map(h => `${h}:00`).join(", ")}</td>
                    <td style={{ padding: "11px 14px", fontSize: "13px", textAlign: "center" }}>{n.avg_ndvi.toFixed(3)}</td>
                    <td style={{ padding: "11px 14px", fontSize: "13px", textAlign: "center" }}>{(n.avg_human_distance / 1000).toFixed(1)} km</td>
                    <td style={{ padding: "11px 14px", textAlign: "center" }}>
                      <span style={{ fontSize: "16px" }}>{n.protected ? "🌿" : "🏘️"}</span>
                    </td>
                    <td style={{ padding: "11px 14px" }}>
                      <span style={{ backgroundColor: sb, color: sc, padding: "3px 10px", borderRadius: "10px", fontSize: "11px", fontWeight: "700", whiteSpace: "nowrap" }}>
                        {safetyLabel(n.safety_score)} · {n.safety_score.toFixed(0)}
                      </span>
                    </td>
                  </tr>
                  {isExp && (
                    <tr key={`exp-${n.node_id}`} style={{ backgroundColor: "#f1f8e9" }}>
                      <td colSpan={10} style={{ padding: "14px 24px" }}>
                        <div style={{ display: "flex", gap: "24px", flexWrap: "wrap", fontSize: "12px", color: "#444" }}>
                          <div><strong>Radius:</strong> {n.radius_meters.toFixed(0)} m</div>
                          <div><strong>Safety Score:</strong> {n.safety_score.toFixed(1)} / 100</div>
                          <div><strong>Land Cover:</strong> {n.land_cover}</div>
                          {n.elephants && n.elephants.filter(e => e !== "Negative").length > 0 && (
                            <div><strong>Known Elephants:</strong>{" "}
                              {n.elephants.filter(e => e !== "Negative").map((e, i) => (
                                <span key={i} style={{ display: "inline-block", backgroundColor: "#c8e6c9", color: "#1B5E20", padding: "2px 8px", borderRadius: "10px", margin: "0 3px", fontWeight: "600" }}>{e}</span>
                              ))}
                            </div>
                          )}
                        </div>
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
          <div style={{ textAlign: "center", padding: "40px", color: "#888", fontSize: "15px" }}>No hotspots match the current filters.</div>
        )}
      </div>
    </div>
  );
}
