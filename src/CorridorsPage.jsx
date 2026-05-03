import { useState, useEffect } from "react";
import axios from "axios";
import { DISHAN_API } from "./apiConfig";
import { Search, ChevronDown, ChevronRight, RefreshCw } from "lucide-react";

const PAGE_SIZE = 20;

const safetyLabel = (s) => s < 30 ? "High" : s < 60 ? "Medium" : s < 80 ? "Moderate" : "Low";
const safetyColor = (s) => s < 30 ? "#ef4444" : s < 60 ? "#f59e0b" : s < 80 ? "#fbbf24" : "#10b981";
const safetyBg    = (s) => s < 30 ? "rgba(239,68,68,0.15)" : s < 60 ? "rgba(245,158,11,0.15)" : s < 80 ? "rgba(251,191,36,0.15)" : "rgba(16,185,129,0.15)";

const G = {
  bg: "#0a0e14",
  card: "rgba(255,255,255,0.06)",
  border: "rgba(255,255,255,0.10)",
  inputBg: "rgba(255,255,255,0.07)",
  textPrimary: "#f9fafb",
  textSecondary: "rgba(255,255,255,0.65)",
  textMuted: "rgba(255,255,255,0.38)",
};

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
    axios.get(`${DISHAN_API}/corridors`)
      .then(r => { setCorridors(r.data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const filtered = corridors.filter(c => {
    if (search && !c.corridor_id.includes(search)) return false;
    if (dangerFilter !== "All" && safetyLabel(c.safety_score) !== dangerFilter) return false;
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === "safety_asc")  return a.safety_score - b.safety_score;
    if (sortBy === "safety_desc") return b.safety_score - a.safety_score;
    if (sortBy === "usage")       return b.usage_count - a.usage_count;
    if (sortBy === "crossings")   return b.crossing_count - a.crossing_count;
    if (sortBy === "distance")    return b.distance_meters - a.distance_meters;
    return 0;
  });

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
  const paged = sorted.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

  const selectStyle = {
    padding: "9px 14px",
    fontSize: "13px",
    borderRadius: "10px",
    border: `1px solid ${G.border}`,
    background: G.inputBg,
    color: G.textPrimary,
    cursor: "pointer",
    outline: "none",
    backdropFilter: "blur(10px)",
  };

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", background: G.bg, color: G.textSecondary, fontSize: "16px", fontFamily: "'Inter', sans-serif" }}>
      <RefreshCw size={20} style={{ marginRight: "12px", animation: "spin 1s linear infinite" }} />
      Loading corridors…
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
  if (error) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 60px)", background: G.bg, color: "#ef4444", fontFamily: "'Inter', sans-serif" }}>
      ⚠️ {error}. Make sure the API server is running.
    </div>
  );

  return (
    <div style={{ minHeight: "calc(100vh - 60px)", background: G.bg, fontFamily: "'Inter', -apple-system, sans-serif" }}>

      {/* Page header */}
      <div style={{
        background: "linear-gradient(135deg, #0a0e14 0%, #064e3b 60%, #0a0e14 100%)",
        padding: "28px 32px 24px",
        borderBottom: `1px solid ${G.border}`,
      }}>
        <h1 style={{
          margin: "0 0 6px 0", fontSize: "24px", fontWeight: "800", letterSpacing: "-0.5px",
          background: "linear-gradient(135deg, #ffffff 0%, #34d399 100%)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text",
        }}>
          🛤️ Elephant Movement Corridors
        </h1>
        <p style={{ margin: 0, fontSize: "13px", color: G.textMuted }}>
          {corridors.length} movement paths between hotspot nodes · Showing {sorted.length} filtered results
        </p>
      </div>

      {/* Summary stats */}
      <div style={{ padding: "16px 32px", display: "flex", gap: "16px", flexWrap: "wrap", borderBottom: `1px solid ${G.border}` }}>
        {[
          { label: "Total Corridors", value: corridors.length,                                                                   color: "#34d399" },
          { label: "High Danger",     value: corridors.filter(c => safetyLabel(c.safety_score) === "High").length,              color: "#ef4444" },
          { label: "Medium Danger",   value: corridors.filter(c => safetyLabel(c.safety_score) === "Medium").length,            color: "#f59e0b" },
          { label: "Avg Safety Score",value: corridors.length ? (corridors.reduce((a, c) => a + c.safety_score, 0) / corridors.length).toFixed(1) : 0, color: "#10b981" },
        ].map(s => (
          <div key={s.label} style={{
            background: G.card, backdropFilter: "blur(20px)", borderRadius: "12px",
            padding: "12px 20px", border: `1px solid ${G.border}`, textAlign: "center", minWidth: "130px",
          }}>
            <div style={{ fontSize: "22px", fontWeight: "800", color: s.color, letterSpacing: "-1px" }}>{s.value}</div>
            <div style={{ fontSize: "11px", color: G.textMuted, marginTop: "3px" }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Filter bar */}
      <div style={{ padding: "14px 32px", display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center", borderBottom: `1px solid ${G.border}`, background: "rgba(255,255,255,0.02)" }}>
        <div style={{ position: "relative" }}>
          <Search size={14} style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", color: G.textMuted }} />
          <input
            placeholder="Search corridor ID (e.g. 33-34)…"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            style={{ ...selectStyle, paddingLeft: "34px", minWidth: "230px" }}
          />
        </div>
        <select value={dangerFilter} onChange={e => { setDanger(e.target.value); setPage(0); }} style={selectStyle}>
          <option value="All">All Danger Levels</option>
          <option value="High">🔴 High Danger</option>
          <option value="Medium">🟠 Medium</option>
          <option value="Moderate">🟡 Moderate</option>
          <option value="Low">🟢 Low</option>
        </select>
        <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={selectStyle}>
          <option value="safety_asc">Sort: Most Dangerous First</option>
          <option value="safety_desc">Sort: Safest First</option>
          <option value="usage">Sort: Most Used</option>
          <option value="crossings">Sort: Most Crossings</option>
          <option value="distance">Sort: Longest First</option>
        </select>
        <span style={{ marginLeft: "auto", fontSize: "13px", color: G.textMuted }}>
          {sorted.length} of {corridors.length} corridors
        </span>
      </div>

      {/* Table */}
      <div style={{ padding: "20px 32px", overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 0, background: G.card, backdropFilter: "blur(20px)", borderRadius: "16px", border: `1px solid ${G.border}`, overflow: "hidden", boxShadow: "0 8px 32px rgba(0,0,0,0.4)" }}>
          <thead>
            <tr style={{ background: "rgba(16,185,129,0.1)", borderBottom: `1px solid ${G.border}` }}>
              {["Corridor ID", "Route", "Length", "Usage", "Crossings", "Active Hours", "Human Dist.", "Road Crossings", "Danger"].map(h => (
                <th key={h} style={{ padding: "13px 14px", textAlign: "left", fontSize: "11px", fontWeight: "700", color: "#34d399", whiteSpace: "nowrap", letterSpacing: "0.5px", textTransform: "uppercase" }}>{h}</th>
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
                    style={{ background: i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent", cursor: "pointer", borderBottom: `1px solid ${G.border}`, transition: "background 0.15s" }}
                    onMouseEnter={e => e.currentTarget.style.background = "rgba(16,185,129,0.06)"}
                    onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent"}
                  >
                    <td style={{ padding: "12px 14px" }}>
                      <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "13px", fontWeight: "700", color: "#34d399" }}>
                        {isExp ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        {c.corridor_id}
                      </span>
                    </td>
                    <td style={{ padding: "12px 14px", fontSize: "12px", color: G.textSecondary, whiteSpace: "nowrap" }}>
                      Node <strong style={{ color: G.textPrimary }}>{c.from_node}</strong> → <strong style={{ color: G.textPrimary }}>{c.to_node}</strong>
                    </td>
                    <td style={{ padding: "12px 14px", fontSize: "13px", textAlign: "center", color: G.textSecondary }}>{(c.distance_meters / 1000).toFixed(1)} km</td>
                    <td style={{ padding: "12px 14px", fontSize: "13px", textAlign: "center", fontWeight: "700", color: G.textPrimary }}>×{c.usage_count}</td>
                    <td style={{ padding: "12px 14px", fontSize: "13px", textAlign: "center", color: G.textSecondary }}>{c.crossing_count}</td>
                    <td style={{ padding: "12px 14px", fontSize: "11px", color: G.textMuted }}>{c.active_hours.slice(0, 4).map(h => `${h}:00`).join(", ")}{c.active_hours.length > 4 ? " …" : ""}</td>
                    <td style={{ padding: "12px 14px", fontSize: "13px", textAlign: "center", color: G.textSecondary }}>{(c.avg_human_distance / 1000).toFixed(1)} km</td>
                    <td style={{ padding: "12px 14px", textAlign: "center" }}>
                      <span style={{ fontSize: "12px", color: G.textSecondary }}>
                        {roadCrossings.length > 0 ? (
                          <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "5px" }}>
                            {roadCrossings.length}
                            {rcHigh > 0 && <span style={{ background: "rgba(239,68,68,0.2)", color: "#ef4444", padding: "1px 6px", borderRadius: "8px", fontSize: "10px", fontWeight: "700", border: "1px solid rgba(239,68,68,0.3)" }}>🔴 {rcHigh} high</span>}
                          </span>
                        ) : <span style={{ color: G.textMuted }}>—</span>}
                      </span>
                    </td>
                    <td style={{ padding: "12px 14px" }}>
                      <span style={{ background: sb, color: sc, padding: "4px 10px", borderRadius: "20px", fontSize: "11px", fontWeight: "700", whiteSpace: "nowrap", border: `1px solid ${sc}40` }}>
                        {safetyLabel(c.safety_score)} · {c.safety_score.toFixed(0)}
                      </span>
                    </td>
                  </tr>

                  {/* Expanded row */}
                  {isExp && (
                    <tr key={`exp-${c.corridor_id}`}>
                      <td colSpan={9} style={{ padding: "16px 24px", background: "rgba(16,185,129,0.04)", borderBottom: `1px solid ${G.border}` }}>
                        <div style={{ display: "flex", gap: "24px", flexWrap: "wrap", fontSize: "12px", color: G.textSecondary, marginBottom: "12px" }}>
                          <div><strong style={{ color: G.textPrimary }}>Bidirectional:</strong> {c.bidirectional ? "Yes" : "No"}</div>
                          <div><strong style={{ color: G.textPrimary }}>Safety Score:</strong> {c.safety_score.toFixed(1)} / 100</div>
                          <div><strong style={{ color: G.textPrimary }}>Elephants Used:</strong> {c.elephants ? c.elephants.join(", ") : "—"}</div>
                        </div>
                        {roadCrossings.length > 0 ? (
                          <>
                            <div style={{ fontWeight: "700", marginBottom: "10px", fontSize: "12px", color: "#34d399" }}>🚗 Road Crossings along this corridor:</div>
                            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                              {roadCrossings.slice(0, 8).map((rc, j) => {
                                const dc = rc.danger_level === "High" ? "#ef4444" : rc.danger_level === "Medium" ? "#f59e0b" : "#fbbf24";
                                return (
                                  <div key={j} style={{ background: G.card, border: `1px solid ${dc}50`, borderRadius: "10px", padding: "10px 14px", fontSize: "11px", minWidth: "165px", backdropFilter: "blur(10px)" }}>
                                    <div style={{ fontWeight: "700", color: dc, marginBottom: "6px" }}>{rc.danger_level} · Score {rc.danger_score.toFixed(0)}</div>
                                    <div style={{ color: G.textSecondary }}>{rc.road_type}</div>
                                    <div style={{ color: G.textMuted }}>🌙 Night: {(rc.night_ratio * 100).toFixed(0)}%</div>
                                    <div style={{ color: G.textMuted }}>🌿 Season: {rc.peak_season}</div>
                                    <div style={{ color: G.textMuted }}>⏰ Peak: {rc.peak_hours.map(h => `${h}:00`).join(", ")}</div>
                                  </div>
                                );
                              })}
                              {roadCrossings.length > 8 && (
                                <div style={{ display: "flex", alignItems: "center", fontSize: "11px", color: G.textMuted }}>+{roadCrossings.length - 8} more…</div>
                              )}
                            </div>
                          </>
                        ) : (
                          <div style={{ fontSize: "12px", color: G.textMuted }}>No road crossings detected along this corridor.</div>
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
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "8px", marginTop: "24px" }}>
            {[
              { label: "← Prev", disabled: page === 0, action: () => setPage(p => p - 1) },
            ].map(btn => (
              <button key={btn.label} disabled={btn.disabled} onClick={btn.action} style={{ padding: "8px 18px", borderRadius: "10px", border: `1px solid ${G.border}`, cursor: btn.disabled ? "not-allowed" : "pointer", background: btn.disabled ? "rgba(255,255,255,0.03)" : G.inputBg, color: btn.disabled ? G.textMuted : G.textPrimary, fontSize: "13px", fontWeight: "500", backdropFilter: "blur(10px)", transition: "all 0.2s" }}>
                {btn.label}
              </button>
            ))}
            <span style={{ fontSize: "13px", color: G.textSecondary, padding: "0 8px" }}>Page {page + 1} of {totalPages}</span>
            <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)} style={{ padding: "8px 18px", borderRadius: "10px", border: `1px solid ${G.border}`, cursor: page >= totalPages - 1 ? "not-allowed" : "pointer", background: page >= totalPages - 1 ? "rgba(255,255,255,0.03)" : G.inputBg, color: page >= totalPages - 1 ? G.textMuted : G.textPrimary, fontSize: "13px", fontWeight: "500", backdropFilter: "blur(10px)" }}>
              Next →
            </button>
          </div>
        )}
        {paged.length === 0 && (
          <div style={{ textAlign: "center", padding: "48px", color: G.textMuted, fontSize: "15px" }}>No corridors match the current filters.</div>
        )}
      </div>
    </div>
  );
}
