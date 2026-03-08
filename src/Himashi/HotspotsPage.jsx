import { useState, useEffect } from "react";
import axios from "axios";
import { DISHAN_API } from "../apiConfig";
import { Search, Filter, AlertTriangle, MapPin, Loader2 } from "lucide-react";

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

const dangerStyle = (s) => {
  if (s < 30) return { bg: "rgba(239,68,68,0.12)", color: "#f87171", border: "rgba(239,68,68,0.3)" };
  if (s < 60) return { bg: "rgba(245,158,11,0.12)", color: "#fbbf24", border: "rgba(245,158,11,0.3)" };
  if (s < 80) return { bg: "rgba(251,191,36,0.1)", color: "#fcd34d", border: "rgba(251,191,36,0.25)" };
  return { bg: "rgba(16,185,129,0.12)", color: "#34d399", border: "rgba(16,185,129,0.3)" };
};

const selectStyle = {
  padding: "10px 14px",
  fontSize: "13px",
  borderRadius: "10px",
  border: "1px solid rgba(255,255,255,0.15)",
  cursor: "pointer",
  outline: "none",
  background: "rgba(255,255,255,0.06)",
  color: "#e5e7eb",
  fontFamily: "'Inter', -apple-system, sans-serif",
  backdropFilter: "blur(10px)",
};

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

  if (loading) return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "center",
      height: "calc(100vh - 60px)", fontSize: "16px", color: "#34d399",
      background: "linear-gradient(135deg, #0a0e14 0%, #064e3b 50%, #0a0e14 100%)",
      fontFamily: "'Inter', sans-serif", gap: "12px"
    }}>
      <Loader2 size={24} style={{ animation: "spin 1s linear infinite" }} />
      Loading hotspots...
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  if (error) return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "center",
      height: "calc(100vh - 60px)", color: "#f87171", gap: "10px",
      background: "linear-gradient(135deg, #0a0e14 0%, #064e3b 50%, #0a0e14 100%)",
      fontFamily: "'Inter', sans-serif", fontSize: "15px"
    }}>
      <AlertTriangle size={20} />
      {error}. Make sure API server is running.
    </div>
  );

  return (
    <div style={{
      minHeight: "calc(100vh - 60px)",
      background: "linear-gradient(135deg, #0a0e14 0%, #064e3b 50%, #0a0e14 100%)",
      backgroundAttachment: "fixed",
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    }}>

      {/* Page header */}
      <div style={{
        background: "rgba(255,255,255,0.06)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid rgba(255,255,255,0.1)",
        padding: "28px 36px",
        display: "flex",
        alignItems: "center",
        gap: "14px",
      }}>
        <div style={{
          width: 48, height: 48, borderRadius: "14px",
          background: "rgba(16,185,129,0.15)",
          border: "1px solid rgba(16,185,129,0.3)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "24px"
        }}>🐘</div>
        <div>
          <h1 style={{ margin: "0 0 4px", fontSize: "22px", fontWeight: "800", color: "#f3f4f6" }}>
            Elephant Hotspot Zones
          </h1>
          <p style={{ margin: 0, fontSize: "13px", color: "#9ca3af" }}>
            {nodes.length} high-use elephant zones detected via DBSCAN clustering · Showing {sorted.length} filtered results
          </p>
        </div>
      </div>

      {/* Filter bar */}
      <div style={{
        background: "rgba(255,255,255,0.04)",
        backdropFilter: "blur(10px)",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        padding: "14px 36px",
        display: "flex",
        flexWrap: "wrap",
        gap: "10px",
        alignItems: "center"
      }}>
        <div style={{ position: "relative" }}>
          <Search size={14} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "#6b7280" }} />
          <input
            placeholder="Search node ID or district..."
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            style={{ ...selectStyle, paddingLeft: "34px", minWidth: "220px" }}
          />
        </div>
        <select value={districtFilter} onChange={e => { setDistrict(e.target.value); setPage(0); }} style={selectStyle}>
          {DISTRICTS.map(d => <option key={d} value={d} style={{ background: "#1f2937" }}>{d === "All" ? "All Districts" : d}</option>)}
        </select>
        <select value={dangerFilter} onChange={e => { setDanger(e.target.value); setPage(0); }} style={selectStyle}>
          <option value="All" style={{ background: "#1f2937" }}>All Danger Levels</option>
          <option value="High" style={{ background: "#1f2937" }}>High Danger</option>
          <option value="Medium" style={{ background: "#1f2937" }}>Medium</option>
          <option value="Moderate" style={{ background: "#1f2937" }}>Moderate</option>
          <option value="Low" style={{ background: "#1f2937" }}>Low</option>
        </select>
        <select value={eleFilter} onChange={e => { setEle(e.target.value); setPage(0); }} style={selectStyle}>
          <option value="All" style={{ background: "#1f2937" }}>All Elephant Counts</option>
          <option value="1" style={{ background: "#1f2937" }}>1 Elephant</option>
          <option value="2" style={{ background: "#1f2937" }}>2 Elephants</option>
          <option value="3+" style={{ background: "#1f2937" }}>3+ Elephants</option>
        </select>
        <select value={protectedFilter} onChange={e => { setProtected(e.target.value); setPage(0); }} style={selectStyle}>
          <option value="All" style={{ background: "#1f2937" }}>All Areas</option>
          <option value="1" style={{ background: "#1f2937" }}>Protected Only</option>
          <option value="0" style={{ background: "#1f2937" }}>Unprotected Only</option>
        </select>
        <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={selectStyle}>
          <option value="safety_asc" style={{ background: "#1f2937" }}>Most Dangerous First</option>
          <option value="safety_desc" style={{ background: "#1f2937" }}>Safest First</option>
          <option value="sightings" style={{ background: "#1f2937" }}>Most Sightings</option>
          <option value="elephants" style={{ background: "#1f2937" }}>Most Elephants</option>
          <option value="human_dist" style={{ background: "#1f2937" }}>Closest to Humans</option>
          <option value="id" style={{ background: "#1f2937" }}>Node ID</option>
        </select>
        <span style={{ marginLeft: "auto", fontSize: "12px", color: "#6b7280", display: "flex", alignItems: "center", gap: "6px" }}>
          <Filter size={13} />
          {sorted.length} of {nodes.length} zones
        </span>
      </div>

      {/* Table */}
      <div style={{ padding: "24px 36px", overflowX: "auto" }}>
        <div style={{
          background: "rgba(255,255,255,0.06)",
          backdropFilter: "blur(20px)",
          borderRadius: "20px",
          border: "1px solid rgba(255,255,255,0.12)",
          overflow: "hidden",
          boxShadow: "0 8px 32px rgba(0,0,0,0.37)"
        }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "rgba(16,185,129,0.08)", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                {["Node ID", "District", "Coordinates", "Elephants", "Sightings", "Active Hours", "NDVI", "Human Dist.", "Protected", "Danger"].map(h => (
                  <th key={h} style={{
                    padding: "14px 16px",
                    textAlign: "left",
                    fontSize: "11px",
                    fontWeight: "700",
                    color: "#9ca3af",
                    whiteSpace: "nowrap",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px"
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paged.map((n, i) => {
                const isExp = expanded === n.node_id;
                const ds = dangerStyle(n.safety_score);
                return (
                  <>
                    <tr
                      key={n.node_id}
                      onClick={() => setExpanded(isExp ? null : n.node_id)}
                      style={{
                        backgroundColor: i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent",
                        cursor: "pointer",
                        borderBottom: "1px solid rgba(255,255,255,0.05)",
                        transition: "background 0.15s"
                      }}
                      onMouseEnter={e => e.currentTarget.style.backgroundColor = "rgba(16,185,129,0.07)"}
                      onMouseLeave={e => e.currentTarget.style.backgroundColor = i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent"}
                    >
                      <td style={{ padding: "12px 16px", fontSize: "13px", fontWeight: "700", color: "#34d399" }}>
                        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                          <MapPin size={13} color="#10b981" />
                          #{n.node_id}
                        </span>
                      </td>
                      <td style={{ padding: "12px 16px", fontSize: "13px", color: "#e5e7eb" }}>{n.district}</td>
                      <td style={{ padding: "12px 16px", fontSize: "11px", color: "#6b7280", fontFamily: "monospace" }}>
                        {n.center_lat.toFixed(4)}, {n.center_lon.toFixed(4)}
                      </td>
                      <td style={{ padding: "12px 16px", fontSize: "13px", textAlign: "center" }}>
                        {"🐘".repeat(Math.min(n.elephant_count, 5))}
                        <span style={{ fontSize: "11px", color: "#6b7280", marginLeft: "4px" }}>×{n.elephant_count}</span>
                      </td>
                      <td style={{ padding: "12px 16px", fontSize: "13px", textAlign: "center", fontWeight: "700", color: "#e5e7eb" }}>{n.sighting_count}</td>
                      <td style={{ padding: "12px 16px", fontSize: "11px", color: "#9ca3af" }}>{n.active_hours.map(h => `${h}:00`).join(", ")}</td>
                      <td style={{ padding: "12px 16px", fontSize: "13px", textAlign: "center", color: "#e5e7eb" }}>{n.avg_ndvi.toFixed(3)}</td>
                      <td style={{ padding: "12px 16px", fontSize: "13px", textAlign: "center", color: "#e5e7eb" }}>{(n.avg_human_distance / 1000).toFixed(1)} km</td>
                      <td style={{ padding: "12px 16px", textAlign: "center" }}>
                        <span style={{ fontSize: "16px" }}>{n.protected ? "🌿" : "🏘️"}</span>
                      </td>
                      <td style={{ padding: "12px 16px" }}>
                        <span style={{
                          backgroundColor: ds.bg,
                          color: ds.color,
                          border: `1px solid ${ds.border}`,
                          padding: "4px 12px",
                          borderRadius: "20px",
                          fontSize: "11px",
                          fontWeight: "700",
                          whiteSpace: "nowrap",
                          display: "inline-block"
                        }}>
                          {safetyLabel(n.safety_score)} · {n.safety_score.toFixed(0)}
                        </span>
                      </td>
                    </tr>
                    {isExp && (
                      <tr key={`exp-${n.node_id}`}>
                        <td colSpan={10} style={{
                          padding: "16px 28px",
                          background: "rgba(16,185,129,0.05)",
                          borderBottom: "1px solid rgba(255,255,255,0.06)"
                        }}>
                          <div style={{ display: "flex", gap: "24px", flexWrap: "wrap", fontSize: "12px", color: "#9ca3af" }}>
                            <div><strong style={{ color: "#34d399" }}>Radius:</strong> {n.radius_meters.toFixed(0)} m</div>
                            <div><strong style={{ color: "#34d399" }}>Safety Score:</strong> {n.safety_score.toFixed(1)} / 100</div>
                            <div><strong style={{ color: "#34d399" }}>Land Cover:</strong> {n.land_cover}</div>
                            {n.elephants && n.elephants.filter(e => e !== "Negative").length > 0 && (
                              <div>
                                <strong style={{ color: "#34d399" }}>Known Elephants:</strong>{" "}
                                {n.elephants.filter(e => e !== "Negative").map((e, idx) => (
                                  <span key={idx} style={{
                                    display: "inline-block",
                                    background: "rgba(16,185,129,0.15)",
                                    color: "#34d399",
                                    border: "1px solid rgba(16,185,129,0.3)",
                                    padding: "2px 10px",
                                    borderRadius: "20px",
                                    margin: "0 3px",
                                    fontWeight: "700"
                                  }}>{e}</span>
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
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "10px", marginTop: "24px" }}>
            <button
              disabled={page === 0}
              onClick={() => setPage(p => p - 1)}
              style={{
                padding: "10px 20px",
                borderRadius: "10px",
                border: "1px solid rgba(255,255,255,0.15)",
                cursor: page === 0 ? "not-allowed" : "pointer",
                background: page === 0 ? "rgba(255,255,255,0.03)" : "rgba(16,185,129,0.12)",
                color: page === 0 ? "#6b7280" : "#34d399",
                fontSize: "13px",
                fontWeight: "600",
                fontFamily: "'Inter', sans-serif",
                transition: "all 0.2s",
              }}
            >← Prev</button>
            <span style={{ fontSize: "13px", color: "#9ca3af", background: "rgba(255,255,255,0.06)", padding: "10px 20px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.1)" }}>
              Page {page + 1} of {totalPages}
            </span>
            <button
              disabled={page >= totalPages - 1}
              onClick={() => setPage(p => p + 1)}
              style={{
                padding: "10px 20px",
                borderRadius: "10px",
                border: "1px solid rgba(255,255,255,0.15)",
                cursor: page >= totalPages - 1 ? "not-allowed" : "pointer",
                background: page >= totalPages - 1 ? "rgba(255,255,255,0.03)" : "rgba(16,185,129,0.12)",
                color: page >= totalPages - 1 ? "#6b7280" : "#34d399",
                fontSize: "13px",
                fontWeight: "600",
                fontFamily: "'Inter', sans-serif",
                transition: "all 0.2s",
              }}
            >Next →</button>
          </div>
        )}
        {paged.length === 0 && (
          <div style={{ textAlign: "center", padding: "60px", color: "#6b7280", fontSize: "15px" }}>
            No hotspots match the current filters.
          </div>
        )}
      </div>
    </div>
  );
}
