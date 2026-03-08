import React, { useState, useEffect, useRef, useCallback } from "react";
import { MapContainer, TileLayer, Polyline, CircleMarker, Popup, useMap, useMapEvents } from "react-leaflet";
import axios from "axios";
import "leaflet/dist/leaflet.css";
import { DISHAN_API as API } from "./apiConfig";
import { X, SlidersHorizontal, RefreshCw } from "lucide-react";

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

function haversine(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a = Math.sin(dLat / 2) ** 2
          + Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180)
          * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function closestPointOnWay(lat, lng, coords) {
  let minDist = Infinity, closestPt = null;
  coords.forEach(([wlat, wlng]) => {
    const d = haversine(lat, lng, wlat, wlng);
    if (d < minDist) { minDist = d; closestPt = [wlat, wlng]; }
  });
  return { point: closestPt, dist: minDist };
}

function findNearestRoad(lat, lng, roads, maxKm = 0.5) {
  let best = null, bestDist = Infinity, bestPt = null;
  roads.forEach(road => {
    const { point, dist } = closestPointOnWay(lat, lng, road.coords);
    if (dist < bestDist && dist <= maxKm) { best = road; bestDist = dist; bestPt = point; }
  });
  return best ? { road: best, closestPoint: bestPt, distance: bestDist } : null;
}

const ROAD_HIGHWAY_TYPES = "primary|secondary|tertiary|trunk|motorway|residential|unclassified";

function RoadFetcher({ onRoads, onFetching }) {
  const map = useMap();
  const timer = useRef(null);
  const cache = useRef({});

  const fetchRoads = useCallback((bounds) => {
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();
    if (ne.lat - sw.lat > 0.15 || ne.lng - sw.lng > 0.15) { onRoads([]); onFetching(false); return; }
    const key = `${sw.lat.toFixed(3)},${sw.lng.toFixed(3)},${ne.lat.toFixed(3)},${ne.lng.toFixed(3)}`;
    if (cache.current[key]) { onRoads(cache.current[key]); return; }
    const query = `[out:json][timeout:30];\nway["highway"~"^(${ROAD_HIGHWAY_TYPES})$"](${sw.lat},${sw.lng},${ne.lat},${ne.lng});\nout geom;`;
    onFetching(true);
    fetch(OVERPASS_URL, { method: "POST", body: `data=${encodeURIComponent(query)}`, headers: { "Content-Type": "application/x-www-form-urlencoded" } })
      .then(r => { if (!r.ok) throw new Error("Overpass error"); return r.json(); })
      .then(data => {
        const roads = (data.elements || [])
          .filter(e => e.type === "way" && Array.isArray(e.geometry) && e.geometry.length >= 2)
          .map(e => ({ id: e.id, highway: e.tags?.highway ?? "unknown", name: e.tags?.name ?? "", coords: e.geometry.map(g => [g.lat, g.lon]) }));
        cache.current[key] = roads;
        onRoads(roads);
        onFetching(false);
      })
      .catch(() => onFetching(false));
  }, [onRoads, onFetching]);

  useMapEvents({
    moveend: () => { clearTimeout(timer.current); timer.current = setTimeout(() => fetchRoads(map.getBounds()), 1200); },
    zoomend: () => { clearTimeout(timer.current); timer.current = setTimeout(() => fetchRoads(map.getBounds()), 1200); },
  });

  useEffect(() => { fetchRoads(map.getBounds()); }, []); // eslint-disable-line
  return null;
}

const DANGER_CONFIG = {
  High:   { color: "#ef4444", weight: 5, opacity: 0.95 },
  Medium: { color: "#f59e0b", weight: 4, opacity: 0.85 },
  Low:    { color: "#fbbf24", weight: 3, opacity: 0.75 },
};

const ROAD_TYPES = ["All", "Major Road", "Minor Road", "Secondary Road", "Other"];
const SEASONS    = ["All", "Dry", "Wet", "Northeast Monsoon", "Southwest Monsoon", "Unknown"];
const TIMES      = ["All", "Morning", "Afternoon", "Evening", "Night"];

function getCorridorDanger(corridor) {
  const crossings = corridor.road_crossings || [];
  if (!crossings.length) return "Low";
  const rank = { High: 3, Medium: 2, Low: 1 };
  return crossings.reduce((best, c) => (rank[c.danger_level] || 0) > (rank[best] || 0) ? c.danger_level : best, "Low");
}

function pathToLatLngs(path) { return (path || []).map(p => [p.lat, p.lon]); }

function topCrossing(corridor) {
  const crossings = corridor.road_crossings || [];
  if (!crossings.length) return null;
  return crossings.reduce((best, c) => (c.danger_score || 0) > (best.danger_score || 0) ? c : best, crossings[0]);
}

function FitBounds({ corridors }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    if (!fitted.current && corridors.length > 0) {
      const allPts = corridors.flatMap(c => pathToLatLngs(c.path));
      if (!allPts.length) return;
      const lats = allPts.map(p => p[0]);
      const lngs = allPts.map(p => p[1]);
      map.fitBounds([[Math.min(...lats), Math.min(...lngs)], [Math.max(...lats), Math.max(...lngs)]], { padding: [40, 40] });
      fitted.current = true;
    }
  }, [corridors, map]);
  return null;
}

const G = {
  panelBg: "rgba(10,14,20,0.96)",
  card: "rgba(255,255,255,0.06)",
  border: "rgba(255,255,255,0.10)",
  inputBg: "rgba(255,255,255,0.08)",
  textPrimary: "#f9fafb",
  textSecondary: "rgba(255,255,255,0.65)",
  textMuted: "rgba(255,255,255,0.38)",
  emerald: "#10b981",
  emeraldLight: "#34d399",
};

const labelStyle = {
  fontSize: "10px", color: G.textMuted, fontWeight: "700",
  textTransform: "uppercase", letterSpacing: "0.6px", display: "block", marginBottom: "5px",
};
const selectStyle = {
  width: "100%", padding: "7px 10px", borderRadius: "8px",
  border: `1px solid ${G.border}`, background: G.inputBg,
  color: G.textPrimary, fontSize: "12px", cursor: "pointer", outline: "none",
};

export default function RoadCrossingsMapPage() {
  const [corridors,    setCorridors]    = useState([]);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState(null);
  const [showPanel,    setShowPanel]    = useState(true);
  const [osmRoads,     setOsmRoads]     = useState([]);
  const [roadFetching, setRoadFetching] = useState(false);
  const [selectedId,   setSelectedId]   = useState(null);

  const [dangers,      setDangers]      = useState({ High: true, Medium: true, Low: true });
  const [roadType,     setRoadType]     = useState("All");
  const [peakSeason,   setPeakSeason]   = useState("All");
  const [peakTime,     setPeakTime]     = useState("All");
  const [minCrossings, setMinCrossings] = useState(1);
  const [minNight,     setMinNight]     = useState(0);

  useEffect(() => {
    axios.get(`${API}/corridors`)
      .then(r => { setCorridors(r.data.filter(c => (c.road_crossings || []).length > 0)); setLoading(false); })
      .catch(() => { setError("Cannot reach API — start the backend."); setLoading(false); });
  }, []);

  const filtered = corridors.filter(corridor => {
    const dangerLvl = getCorridorDanger(corridor);
    if (!dangers[dangerLvl]) return false;
    const crossings = corridor.road_crossings || [];
    if (roadType !== "All" && !crossings.some(c => c.road_type === roadType)) return false;
    if (peakSeason !== "All" && !crossings.some(c => c.peak_season === peakSeason)) return false;
    if (peakTime !== "All" && !crossings.some(c => c.peak_time_of_day === peakTime)) return false;
    const totalCrossings = crossings.reduce((sum, c) => sum + (c.crossing_count || 0), 0);
    if (totalCrossings < minCrossings) return false;
    if (crossings.length > 0) {
      const avgNight = crossings.reduce((s, c) => s + (c.night_ratio || 0), 0) / crossings.length;
      if (avgNight * 100 < minNight) return false;
    }
    return true;
  });

  const counts = { High: 0, Medium: 0, Low: 0 };
  filtered.forEach(c => counts[getCorridorDanger(c)]++);

  const toggleDanger = lvl => setDangers(d => ({ ...d, [lvl]: !d[lvl] }));

  const dangerBadgeStyle = (lvl) => ({
    display: "inline-block", padding: "2px 8px", borderRadius: "20px",
    fontSize: "11px", fontWeight: "700", color: "white",
    background: DANGER_CONFIG[lvl]?.color ?? "#6b7280",
  });

  return (
    <div style={{ position: "relative", height: "calc(100vh - 60px)", fontFamily: "'Inter', -apple-system, sans-serif" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* Toggle button when panel hidden */}
      {!showPanel && (
        <button onClick={() => setShowPanel(true)} style={{
          position: "absolute", top: "12px", left: "12px", zIndex: 1001,
          padding: "9px 16px", background: G.panelBg, backdropFilter: "blur(20px)",
          color: G.textPrimary, border: `1px solid ${G.border}`,
          borderRadius: "10px", cursor: "pointer", fontSize: "13px", fontWeight: "600",
          boxShadow: "0 4px 16px rgba(0,0,0,0.5)", display: "flex", alignItems: "center", gap: "7px",
        }}>
          <SlidersHorizontal size={14} color={G.emeraldLight} /> Filters
        </button>
      )}

      {/* Stats bar */}
      <div style={{
        position: "absolute", top: "12px", right: "12px", zIndex: 1000,
        background: G.panelBg, backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
        borderRadius: "12px", boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
        padding: "10px 16px", display: "flex", gap: "16px", alignItems: "center",
        fontSize: "12px", border: `1px solid ${G.border}`,
      }}>
        <span style={{ fontWeight: "700", color: G.textPrimary }}>
          {filtered.length.toLocaleString()} corridors
        </span>
        <span style={{ color: "#ef4444", fontWeight: "600" }}>🔴 {counts.High} High</span>
        <span style={{ color: "#f59e0b", fontWeight: "600" }}>🟠 {counts.Medium} Med</span>
        <span style={{ color: "#fbbf24", fontWeight: "600" }}>🟡 {counts.Low} Low</span>
        {roadFetching && (
          <span style={{ color: G.emeraldLight, display: "flex", alignItems: "center", gap: "5px" }}>
            <RefreshCw size={11} style={{ animation: "spin 1s linear infinite" }} /> loading roads…
          </span>
        )}
        {!roadFetching && osmRoads.length === 0 && (
          <span style={{ color: G.textMuted, fontStyle: "italic" }}>🔍 Zoom in to match roads</span>
        )}
        {!roadFetching && osmRoads.length > 0 && (
          <span style={{ color: G.emeraldLight }}>🛣️ {osmRoads.length} roads</span>
        )}
      </div>

      {/* Filter Panel */}
      {showPanel && (
        <div style={{
          position: "absolute", top: "12px", left: "12px", zIndex: 1000,
          width: "260px", background: G.panelBg,
          backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
          borderRadius: "16px", boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
          overflow: "hidden", border: `1px solid ${G.border}`,
        }}>
          {/* Panel header */}
          <div style={{
            background: "rgba(16,185,129,0.12)", borderBottom: `1px solid ${G.border}`,
            padding: "13px 16px", display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <span style={{ fontWeight: "700", fontSize: "13px", color: G.emeraldLight, display: "flex", alignItems: "center", gap: "6px" }}>
              🚗 Corridor-Road Overlaps
            </span>
            <button onClick={() => setShowPanel(false)} style={{
              background: "none", border: "none", color: G.textMuted,
              cursor: "pointer", padding: "2px", display: "flex", alignItems: "center",
              transition: "color 0.2s",
            }}
              onMouseEnter={e => e.currentTarget.style.color = G.textPrimary}
              onMouseLeave={e => e.currentTarget.style.color = G.textMuted}
            >
              <X size={16} />
            </button>
          </div>

          {/* Loading / Error */}
          {loading && (
            <div style={{ padding: "16px", fontSize: "12px", color: G.textMuted, display: "flex", alignItems: "center", gap: "8px" }}>
              <RefreshCw size={13} style={{ animation: "spin 1s linear infinite", color: G.emerald }} />
              Loading crossings…
            </div>
          )}
          {error && <div style={{ padding: "14px", fontSize: "12px", color: "#f87171" }}>{error}</div>}

          {/* Danger toggles */}
          <div style={{ padding: "12px 14px", borderBottom: `1px solid ${G.border}` }}>
            <span style={labelStyle}>Danger Level</span>
            {["High", "Medium", "Low"].map(lvl => (
              <label key={lvl} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px", cursor: "pointer" }}>
                <input
                  type="checkbox" checked={dangers[lvl]}
                  onChange={() => toggleDanger(lvl)}
                  style={{ accentColor: DANGER_CONFIG[lvl].color, width: "14px", height: "14px" }}
                />
                <span style={dangerBadgeStyle(lvl)}>{lvl}</span>
                <span style={{ fontSize: "12px", color: G.textMuted }}>({counts[lvl]})</span>
              </label>
            ))}
          </div>

          {/* Road type */}
          <div style={{ padding: "10px 14px", borderBottom: `1px solid ${G.border}` }}>
            <span style={labelStyle}>Road Type</span>
            <select value={roadType} onChange={e => setRoadType(e.target.value)} style={selectStyle}>
              {ROAD_TYPES.map(t => <option key={t} style={{ background: "#111827" }}>{t}</option>)}
            </select>
          </div>

          {/* Peak season */}
          <div style={{ padding: "10px 14px", borderBottom: `1px solid ${G.border}` }}>
            <span style={labelStyle}>Peak Season</span>
            <select value={peakSeason} onChange={e => setPeakSeason(e.target.value)} style={selectStyle}>
              {SEASONS.map(s => <option key={s} style={{ background: "#111827" }}>{s}</option>)}
            </select>
          </div>

          {/* Peak time */}
          <div style={{ padding: "10px 14px", borderBottom: `1px solid ${G.border}` }}>
            <span style={labelStyle}>Peak Time of Day</span>
            <select value={peakTime} onChange={e => setPeakTime(e.target.value)} style={selectStyle}>
              {TIMES.map(t => <option key={t} style={{ background: "#111827" }}>{t}</option>)}
            </select>
          </div>

          {/* Min crossing events */}
          <div style={{ padding: "10px 14px", borderBottom: `1px solid ${G.border}` }}>
            <span style={labelStyle}>Min Crossing Events: <strong style={{ color: G.textPrimary }}>{minCrossings}</strong></span>
            <input type="range" min={1} max={100} value={minCrossings}
              onChange={e => setMinCrossings(Number(e.target.value))}
              style={{ width: "100%", accentColor: G.emerald }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", color: G.textMuted }}>
              <span>1</span><span>100</span>
            </div>
          </div>

          {/* Min night ratio */}
          <div style={{ padding: "10px 14px", borderBottom: `1px solid ${G.border}` }}>
            <span style={labelStyle}>Min Night Activity: <strong style={{ color: G.textPrimary }}>{minNight}%</strong></span>
            <input type="range" min={0} max={100} value={minNight}
              onChange={e => setMinNight(Number(e.target.value))}
              style={{ width: "100%", accentColor: G.emerald }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", color: G.textMuted }}>
              <span>0%</span><span>100%</span>
            </div>
          </div>

          {/* Reset */}
          <div style={{ padding: "10px 14px" }}>
            <button onClick={() => {
              setDangers({ High: true, Medium: true, Low: true });
              setRoadType("All"); setPeakSeason("All"); setPeakTime("All");
              setMinCrossings(1); setMinNight(0);
            }} style={{
              width: "100%", padding: "8px", background: "rgba(255,255,255,0.06)",
              border: `1px solid ${G.border}`, borderRadius: "8px", cursor: "pointer",
              fontSize: "12px", color: G.textSecondary, fontWeight: "600",
              transition: "all 0.2s",
            }}
              onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.1)"}
              onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.06)"}
            >
              ↺ Reset Filters
            </button>
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{
        position: "absolute", bottom: "20px", left: "12px", zIndex: 1000,
        background: G.panelBg, backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
        borderRadius: "14px", boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
        padding: "12px 16px", border: `1px solid ${G.border}`, minWidth: "175px",
      }}>
        <div style={{ fontWeight: "700", fontSize: "11px", color: G.emeraldLight, marginBottom: "10px", textTransform: "uppercase", letterSpacing: "0.5px" }}>
          Corridor Danger Level
        </div>
        {Object.entries(DANGER_CONFIG).map(([lvl, cfg]) => (
          <div key={lvl} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
            <svg width="28" height="10">
              <line x1="0" y1="5" x2="28" y2="5" stroke={cfg.color} strokeWidth={cfg.weight} strokeLinecap="round" />
            </svg>
            <span style={{ fontSize: "12px", color: G.textSecondary }}>{lvl} risk corridor</span>
          </div>
        ))}
        <div style={{ marginTop: "10px", paddingTop: "10px", borderTop: `1px solid ${G.border}`, fontSize: "10px", color: G.textMuted }}>
          Click a corridor to show matched road
        </div>
        <div style={{ marginTop: "6px", display: "flex", alignItems: "center", gap: "6px" }}>
          <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="#60a5fa" strokeWidth="3" /></svg>
          <span style={{ fontSize: "11px", color: G.textSecondary }}>Matched Road</span>
        </div>
        <div style={{ marginTop: "5px", display: "flex", alignItems: "center", gap: "6px" }}>
          <svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#6366f1" fillOpacity="0.8" /></svg>
          <span style={{ fontSize: "11px", color: G.textSecondary }}>Crossing point</span>
        </div>
      </div>

      {/* Map */}
      <MapContainer center={[7.8731, 80.7718]} zoom={8}
        style={{ height: "100%", width: "100%" }}
        maxBounds={[[5.5, 79.0], [10.2, 82.5]]} minZoom={7} maxZoom={16} maxBoundsViscosity={0.8}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; OpenStreetMap contributors' />
        <FitBounds corridors={filtered} />
        <RoadFetcher onRoads={setOsmRoads} onFetching={setRoadFetching} />

        {filtered.map((corridor) => {
          const dangerLvl  = getCorridorDanger(corridor);
          const cfg        = DANGER_CONFIG[dangerLvl] || DANGER_CONFIG.Low;
          const positions  = pathToLatLngs(corridor.path);
          const isSelected = selectedId === corridor.corridor_id;
          const top        = topCrossing(corridor);
          const match = isSelected && top && osmRoads.length > 0
            ? findNearestRoad(top.crossing_lat, top.crossing_lon, osmRoads)
            : null;

          return (
            <React.Fragment key={corridor.corridor_id}>
              {isSelected && match && (
                <Polyline positions={match.road.coords}
                  pathOptions={{ color: "#60a5fa", weight: 6, opacity: 0.9 }}
                >
                  <Popup maxWidth={200}>
                    <div style={{ fontFamily: "'Inter', sans-serif", fontSize: "12px" }}>
                      <strong>🛣️ {match.road.name || match.road.highway}</strong>
                      <div style={{ color: "#666", marginTop: "4px" }}>OSM ID: {match.road.id}</div>
                      <div style={{ color: "#666" }}>Type: {match.road.highway}</div>
                    </div>
                  </Popup>
                </Polyline>
              )}

              {isSelected && (corridor.road_crossings || []).map((cr, ci) => (
                <CircleMarker key={ci} center={[cr.crossing_lat, cr.crossing_lon]} radius={6}
                  pathOptions={{ color: "#6366f1", fillColor: "#818cf8", fillOpacity: 0.85, weight: 2 }}
                >
                  <Popup maxWidth={220}>
                    <div style={{ fontFamily: "'Inter', sans-serif", fontSize: "12px", lineHeight: 1.5 }}>
                      <strong>Road Crossing</strong>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3px 8px", marginTop: "6px" }}>
                        <span style={{ color: "#888" }}>Road Type</span><span>{cr.road_type ?? "—"}</span>
                        <span style={{ color: "#888" }}>Events</span><span style={{ fontWeight: "600" }}>{cr.crossing_count}</span>
                        <span style={{ color: "#888" }}>Danger</span><span style={{ fontWeight: "700", color: cfg.color }}>{cr.danger_level}</span>
                        <span style={{ color: "#888" }}>Night</span><span>🌙 {cr.night_ratio != null ? (cr.night_ratio * 100).toFixed(0) + "%" : "—"}</span>
                        <span style={{ color: "#888" }}>Peak Time</span><span>{cr.peak_time_of_day ?? "—"}</span>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}

              <Polyline positions={positions}
                pathOptions={{
                  color:     isSelected ? "#6366f1" : cfg.color,
                  weight:    isSelected ? cfg.weight + 2 : cfg.weight,
                  opacity:   isSelected ? 1 : cfg.opacity,
                  dashArray: isSelected ? null : "8,4",
                }}
                eventHandlers={{ click: () => setSelectedId(isSelected ? null : corridor.corridor_id) }}
              >
                <Popup maxWidth={300}>
                  <div style={{ fontFamily: "'Inter', sans-serif", lineHeight: 1.5 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px", paddingBottom: "8px", borderBottom: "1px solid #eee" }}>
                      <strong style={{ fontSize: "13px", color: "#222" }}>Corridor {corridor.corridor_id}</strong>
                      <span style={{ padding: "2px 8px", borderRadius: "10px", fontSize: "11px", fontWeight: "700", color: "white", background: cfg.color }}>{dangerLvl}</span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 10px", fontSize: "12px" }}>
                      <span style={{ color: "#888" }}>Node Path</span><span>{corridor.from_node} → {corridor.to_node}</span>
                      <span style={{ color: "#888" }}>Road Crossings</span><span style={{ fontWeight: "600" }}>{(corridor.road_crossings || []).length}</span>
                      <span style={{ color: "#888" }}>Total Events</span>
                      <span style={{ fontWeight: "600" }}>{(corridor.road_crossings || []).reduce((s, c) => s + (c.crossing_count || 0), 0)}</span>
                      <span style={{ color: "#888" }}>Length</span>
                      <span>{corridor.distance_meters != null ? (corridor.distance_meters / 1000).toFixed(2) + " km" : "—"}</span>
                      <span style={{ color: "#888" }}>Safety Score</span>
                      <span style={{ fontWeight: "600", color: cfg.color }}>{corridor.safety_score?.toFixed(1) ?? "—"}</span>
                      {top && (
                        <>
                          <span style={{ color: "#888" }}>Highest Risk Road</span><span>{top.road_type ?? "—"}</span>
                          <span style={{ color: "#888" }}>Peak Time</span><span>{top.peak_time_of_day ?? "—"}</span>
                          <span style={{ color: "#888" }}>Night Activity</span>
                          <span>🌙 {top.night_ratio != null ? (top.night_ratio * 100).toFixed(0) + "%" : "—"}</span>
                        </>
                      )}
                    </div>
                    <div style={{ marginTop: "6px", paddingTop: "6px", borderTop: "1px solid #eee", fontSize: "10px", color: "#aaa" }}>
                      Click to toggle selection
                    </div>
                  </div>
                </Popup>
              </Polyline>
            </React.Fragment>
          );
        })}
      </MapContainer>
    </div>
  );
}
