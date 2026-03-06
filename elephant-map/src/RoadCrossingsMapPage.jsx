import React, { useState, useEffect, useRef, useCallback } from "react";
import { MapContainer, TileLayer, Polyline, CircleMarker, Popup, useMap, useMapEvents } from "react-leaflet";
import axios from "axios";
import "leaflet/dist/leaflet.css";

const API = "http://localhost:8000";
const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

// ── Haversine distance in km ──────────────────────────────────────────────────
function haversine(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a = Math.sin(dLat / 2) ** 2
          + Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180)
          * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ── Find closest point on a polyline to a given point ────────────────────────
function closestPointOnWay(lat, lng, coords) {
  let minDist = Infinity, closestPt = null;
  coords.forEach(([wlat, wlng]) => {
    const d = haversine(lat, lng, wlat, wlng);
    if (d < minDist) { minDist = d; closestPt = [wlat, wlng]; }
  });
  return { point: closestPt, dist: minDist };
}

// ── Find nearest road from fetched Overpass ways ─────────────────────────────
function findNearestRoad(lat, lng, roads, maxKm = 0.5) {
  let best = null, bestDist = Infinity, bestPt = null;
  roads.forEach(road => {
    const { point, dist } = closestPointOnWay(lat, lng, road.coords);
    if (dist < bestDist && dist <= maxKm) {
      best = road; bestDist = dist; bestPt = point;
    }
  });
  return best ? { road: best, closestPoint: bestPt, distance: bestDist } : null;
}

// ── Fetch real road geometry from Overpass when map moves ────────────────────
// Only fetch main roads to reduce payload and avoid timeouts
const ROAD_HIGHWAY_TYPES = "primary|secondary|tertiary|trunk|motorway|residential|unclassified";

function RoadFetcher({ onRoads, onFetching }) {
  const map = useMap();
  const timer = useRef(null);
  const cache = useRef({}); // cache by bounds key

  const fetchRoads = useCallback((bounds) => {
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();
    const latSpan = ne.lat - sw.lat;
    const lngSpan = ne.lng - sw.lng;

    // Skip if zoomed too far out (require zoom level ~13+)
    if (latSpan > 0.15 || lngSpan > 0.15) {
      onRoads([]);
      onFetching(false);
      return;
    }

    // Round bounds for cache key
    const key = `${sw.lat.toFixed(3)},${sw.lng.toFixed(3)},${ne.lat.toFixed(3)},${ne.lng.toFixed(3)}`;
    if (cache.current[key]) {
      onRoads(cache.current[key]);
      return;
    }

    // Query only main road types (regex)
    const query = `[out:json][timeout:30];
way["highway"~"^(${ROAD_HIGHWAY_TYPES})$"](${sw.lat},${sw.lng},${ne.lat},${ne.lng});
out geom;`;

    onFetching(true);
    fetch(OVERPASS_URL, {
      method: "POST",
      body: `data=${encodeURIComponent(query)}`,
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    })
      .then(r => {
        if (!r.ok) throw new Error("Overpass error");
        return r.json();
      })
      .then(data => {
        const roads = (data.elements || [])
          .filter(e => e.type === "way" && Array.isArray(e.geometry) && e.geometry.length >= 2)
          .map(e => ({
            id: e.id,
            highway: e.tags?.highway ?? "unknown",
            name: e.tags?.name ?? "",
            coords: e.geometry.map(g => [g.lat, g.lon]),
          }));
        cache.current[key] = roads;
        onRoads(roads);
        onFetching(false);
      })
      .catch(() => { onFetching(false); });
  }, [onRoads, onFetching]);

  useMapEvents({
    moveend: () => { clearTimeout(timer.current); timer.current = setTimeout(() => fetchRoads(map.getBounds()), 1200); },
    zoomend: () => { clearTimeout(timer.current); timer.current = setTimeout(() => fetchRoads(map.getBounds()), 1200); },
  });

  useEffect(() => { fetchRoads(map.getBounds()); }, []); // eslint-disable-line
  return null;
}

const DANGER_CONFIG = {
  High:   { color: "#d32f2f", weight: 5, opacity: 0.95 },
  Medium: { color: "#e65100", weight: 4, opacity: 0.85 },
  Low:    { color: "#f9a825", weight: 3, opacity: 0.75 },
};

const ROAD_TYPES = ["All", "Major Road", "Minor Road", "Secondary Road", "Other"];
const SEASONS    = ["All", "Dry", "Wet", "Northeast Monsoon", "Southwest Monsoon", "Unknown"];
const TIMES      = ["All", "Morning", "Afternoon", "Evening", "Night"];

// ── Derive corridor's max danger level from its road crossings ────────────────
function getCorridorDanger(corridor) {
  const crossings = corridor.road_crossings || [];
  if (!crossings.length) return "Low";
  const rank = { High: 3, Medium: 2, Low: 1 };
  return crossings.reduce(
    (best, c) => (rank[c.danger_level] || 0) > (rank[best] || 0) ? c.danger_level : best,
    "Low"
  );
}

// ── Convert corridor path [{lat,lon}] to Leaflet [[lat,lon]] ─────────────────
function pathToLatLngs(path) {
  return (path || []).map(p => [p.lat, p.lon]);
}

// ── Find the most dangerous crossing in a corridor ────────────────────────────
function topCrossing(corridor) {
  const crossings = corridor.road_crossings || [];
  if (!crossings.length) return null;
  return crossings.reduce((best, c) =>
    (c.danger_score || 0) > (best.danger_score || 0) ? c : best
  , crossings[0]);
}

function FitBounds({ corridors }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    if (!fitted.current && corridors.length > 0) {
      const allPts = corridors.flatMap(c => pathToLatLngs(c.path));
      if (allPts.length === 0) return;
      const lats = allPts.map(p => p[0]);
      const lngs = allPts.map(p => p[1]);
      map.fitBounds([
        [Math.min(...lats), Math.min(...lngs)],
        [Math.max(...lats), Math.max(...lngs)],
      ], { padding: [40, 40] });
      fitted.current = true;
    }
  }, [corridors, map]);
  return null;
}

export default function RoadCrossingsMapPage() {
  const [corridors,     setCorridors]     = useState([]);
  const [loading,       setLoading]       = useState(true);
  const [error,         setError]         = useState(null);
  const [showPanel,     setShowPanel]     = useState(true);
  const [osmRoads,      setOsmRoads]      = useState([]);
  const [roadFetching,  setRoadFetching]  = useState(false);
  const [selectedId,    setSelectedId]    = useState(null);

  // Filters
  const [dangers,       setDangers]       = useState({ High: true, Medium: true, Low: true });
  const [roadType,      setRoadType]      = useState("All");
  const [peakSeason,    setPeakSeason]    = useState("All");
  const [peakTime,      setPeakTime]      = useState("All");
  const [minCrossings,  setMinCrossings]  = useState(1);
  const [minNight,      setMinNight]      = useState(0);

  useEffect(() => {
    axios.get(`${API}/corridors`)
      .then(r => {
        // Keep only corridors that actually cross a road
        const withRoads = r.data.filter(c => (c.road_crossings || []).length > 0);
        setCorridors(withRoads);
        setLoading(false);
      })
      .catch(() => { setError("Cannot reach API — start the backend."); setLoading(false); });
  }, []);

  const filtered = corridors.filter(corridor => {
    const dangerLvl = getCorridorDanger(corridor);
    if (!dangers[dangerLvl]) return false;

    const crossings = corridor.road_crossings || [];

    // Road type: at least one crossing must match
    if (roadType !== "All" && !crossings.some(c => c.road_type === roadType)) return false;

    // Peak season: at least one crossing must match
    if (peakSeason !== "All" && !crossings.some(c => c.peak_season === peakSeason)) return false;

    // Peak time: at least one crossing must match
    if (peakTime !== "All" && !crossings.some(c => c.peak_time_of_day === peakTime)) return false;

    // Min crossing events: sum of all crossings on corridor
    const totalCrossings = crossings.reduce((sum, c) => sum + (c.crossing_count || 0), 0);
    if (totalCrossings < minCrossings) return false;

    // Min night ratio: average night ratio across crossings
    if (crossings.length > 0) {
      const avgNight = crossings.reduce((s, c) => s + (c.night_ratio || 0), 0) / crossings.length;
      if (avgNight * 100 < minNight) return false;
    }

    return true;
  });

  const counts = { High: 0, Medium: 0, Low: 0 };
  filtered.forEach(c => counts[getCorridorDanger(c)]++);

  const toggleDanger = lvl => setDangers(d => ({ ...d, [lvl]: !d[lvl] }));

  /* ── Styles ── */
  const panel = {
    position: "absolute", top: "10px", left: "10px", zIndex: 1000,
    width: "270px", backgroundColor: "rgba(255,255,255,0.97)",
    borderRadius: "10px", boxShadow: "0 4px 18px rgba(0,0,0,0.25)",
    overflow: "hidden", border: "1px solid rgba(0,0,0,0.08)",
  };
  const panelHead = {
    background: "linear-gradient(90deg,#0d1b5e,#1a237e)",
    color: "white", padding: "12px 14px",
    display: "flex", justifyContent: "space-between", alignItems: "center",
  };
  const sec = { padding: "12px 14px", borderBottom: "1px solid #f0f0f0" };
  const label = { fontSize: "11px", color: "#888", fontWeight: "600",
                  textTransform: "uppercase", letterSpacing: "0.5px", display: "block", marginBottom: "6px" };
  const select = {
    width: "100%", padding: "5px 8px", borderRadius: "5px",
    border: "1px solid #ddd", fontSize: "12px", color: "#333",
  };
  const badge = (lvl) => ({
    display: "inline-block", padding: "2px 8px", borderRadius: "10px",
    fontSize: "11px", fontWeight: "700", color: "white",
    backgroundColor: lvl === "High" ? "#d32f2f" : lvl === "Medium" ? "#e65100" : "#f9a825",
  });

  return (
    <div style={{ position: "relative", height: "calc(100vh - 60px)" }}>

      {/* Toggle panel button when panel is hidden */}
      {!showPanel && (
        <button onClick={() => setShowPanel(true)} style={{
          position: "absolute", top: "10px", left: "10px", zIndex: 1001,
          padding: "8px 14px", backgroundColor: "#1a237e", color: "white",
          border: "none", borderRadius: "6px", cursor: "pointer", fontSize: "13px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.3)"
        }}>☰ Filters</button>
      )}

      {/* Stats bar */}
      <div style={{
        position: "absolute", top: "10px", right: "10px", zIndex: 1000,
        backgroundColor: "rgba(255,255,255,0.97)", borderRadius: "8px",
        boxShadow: "0 2px 10px rgba(0,0,0,0.2)", padding: "8px 14px",
        display: "flex", gap: "16px", alignItems: "center", fontSize: "12px",
        border: "1px solid rgba(0,0,0,0.08)"
      }}>
        <span style={{ fontWeight: "700", color: "#333" }}>Showing {filtered.length.toLocaleString()} corridors</span>
        <span style={{ color: "#d32f2f", fontWeight: "600" }}>🔴 {counts.High} High</span>
        <span style={{ color: "#e65100", fontWeight: "600" }}>🟠 {counts.Medium} Med</span>
        <span style={{ color: "#f9a825", fontWeight: "600" }}>🟡 {counts.Low} Low</span>
        {roadFetching && (
          <span style={{ color: "#1a237e", fontSize: "11px", fontStyle: "italic" }}>
            ↻ loading roads…
          </span>
        )}
        {!roadFetching && osmRoads.length === 0 && (
          <span style={{ color: "#666", fontSize: "11px", fontStyle: "italic" }}>
            🔍 Zoom in to match roads
          </span>
        )}
        {!roadFetching && osmRoads.length > 0 && (
          <span style={{ color: "#2e7d32", fontSize: "11px" }}>
            🛣️ {osmRoads.length} roads loaded
          </span>
        )}
      </div>

      {/* Filter Panel */}
      {showPanel && (
        <div style={panel}>
          <div style={panelHead}>
            <span style={{ fontWeight: "700", fontSize: "13px" }}>� Corridor-Road Overlaps</span>
            <button onClick={() => setShowPanel(false)} style={{
              background: "none", border: "none", color: "white",
              cursor: "pointer", fontSize: "18px", lineHeight: 1, padding: 0
            }}>×</button>
          </div>

          {/* Loading / Error */}
          {loading && <div style={{ padding: "14px", fontSize: "12px", color: "#666" }}>Loading crossings…</div>}
          {error   && <div style={{ padding: "14px", fontSize: "12px", color: "#d32f2f" }}>{error}</div>}

          {/* Danger level toggles */}
          <div style={sec}>
            <span style={label}>Danger Level</span>
            {["High", "Medium", "Low"].map(lvl => (
              <label key={lvl} style={{
                display: "flex", alignItems: "center", gap: "8px",
                marginBottom: "5px", cursor: "pointer"
              }}>
                <input
                  type="checkbox"
                  checked={dangers[lvl]}
                  onChange={() => toggleDanger(lvl)}
                  style={{ accentColor: DANGER_CONFIG[lvl].fill, width: "15px", height: "15px" }}
                />
                <span style={badge(lvl)}>{lvl}</span>
                <span style={{ fontSize: "12px", color: "#555" }}>({counts[lvl]})</span>
              </label>
            ))}
          </div>

          {/* Road type */}
          <div style={sec}>
            <span style={label}>Road Type</span>
            <select value={roadType} onChange={e => setRoadType(e.target.value)} style={select}>
              {ROAD_TYPES.map(t => <option key={t}>{t}</option>)}
            </select>
          </div>

          {/* Peak season */}
          <div style={sec}>
            <span style={label}>Peak Season</span>
            <select value={peakSeason} onChange={e => setPeakSeason(e.target.value)} style={select}>
              {SEASONS.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>

          {/* Peak time */}
          <div style={sec}>
            <span style={label}>Peak Time of Day</span>
            <select value={peakTime} onChange={e => setPeakTime(e.target.value)} style={select}>
              {TIMES.map(t => <option key={t}>{t}</option>)}
            </select>
          </div>

          {/* Min crossing events */}
          <div style={sec}>
            <span style={label}>Min Crossing Events: <strong>{minCrossings}</strong></span>
            <input
              type="range" min={1} max={100} value={minCrossings}
              onChange={e => setMinCrossings(Number(e.target.value))}
              style={{ width: "100%", accentColor: "#1a237e" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", color: "#aaa" }}>
              <span>1</span><span>100</span>
            </div>
          </div>

          {/* Min night ratio */}
          <div style={sec}>
            <span style={label}>Min Night Activity: <strong>{minNight}%</strong></span>
            <input
              type="range" min={0} max={100} value={minNight}
              onChange={e => setMinNight(Number(e.target.value))}
              style={{ width: "100%", accentColor: "#1a237e" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", color: "#aaa" }}>
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
              width: "100%", padding: "7px", backgroundColor: "#f5f5f5",
              border: "1px solid #ddd", borderRadius: "5px", cursor: "pointer",
              fontSize: "12px", color: "#555", fontWeight: "600"
            }}>↺ Reset Filters</button>
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{
        position: "absolute", bottom: "20px", left: "10px", zIndex: 1000,
        backgroundColor: "rgba(255,255,255,0.97)", borderRadius: "8px",
        boxShadow: "0 2px 10px rgba(0,0,0,0.2)", padding: "10px 14px",
        border: "1px solid rgba(0,0,0,0.08)", minWidth: "180px"
      }}>
        <div style={{ fontWeight: "700", fontSize: "12px", color: "#333", marginBottom: "8px" }}>
          Corridor Danger Level
        </div>
        {Object.entries(DANGER_CONFIG).map(([lvl, cfg]) => (
          <div key={lvl} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "5px" }}>
            <svg width="28" height="10">
              <line x1="0" y1="5" x2="28" y2="5" stroke={cfg.color} strokeWidth={cfg.weight} strokeLinecap="round" />
            </svg>
            <span style={{ fontSize: "12px", color: "#444" }}>{lvl} risk corridor</span>
          </div>
        ))}
        <div style={{ marginTop: "8px", paddingTop: "8px", borderTop: "1px solid #eee",
                      fontSize: "10px", color: "#999" }}>
          Click a corridor to show matched road
        </div>
        <div style={{ marginTop: "6px", display: "flex", alignItems: "center", gap: "6px" }}>
          <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="#2196f3" strokeWidth="3" /></svg>
          <span style={{ fontSize: "11px", color: "#444" }}>Matched Road</span>
        </div>
        <div style={{ marginTop: "4px", display: "flex", alignItems: "center", gap: "6px" }}>
          <svg width="12" height="12">
            <circle cx="6" cy="6" r="5" fill="#1a237e" fillOpacity="0.7" />
          </svg>
          <span style={{ fontSize: "11px", color: "#444" }}>Crossing point</span>
        </div>
      </div>

      {/* Map */}
      <MapContainer
        center={[7.8731, 80.7718]}
        zoom={8}
        style={{ height: "100%", width: "100%" }}
        maxBounds={[[5.5, 79.0], [10.2, 82.5]]}
        minZoom={7}
        maxZoom={16}
        maxBoundsViscosity={0.8}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; OpenStreetMap contributors'
        />
        <FitBounds corridors={filtered} />
        <RoadFetcher onRoads={setOsmRoads} onFetching={setRoadFetching} />

        {filtered.map((corridor) => {
          const dangerLvl  = getCorridorDanger(corridor);
          const cfg        = DANGER_CONFIG[dangerLvl] || DANGER_CONFIG.Low;
          const positions  = pathToLatLngs(corridor.path);
          const isSelected = selectedId === corridor.corridor_id;
          const top        = topCrossing(corridor);

          // Find nearest OSM road to the top crossing point when selected
          const match = isSelected && top && osmRoads.length > 0
            ? findNearestRoad(top.crossing_lat, top.crossing_lon, osmRoads)
            : null;

          return (
            <React.Fragment key={corridor.corridor_id}>
              {/* Highlighted matched road */}
              {isSelected && match && (
                <Polyline
                  positions={match.road.coords}
                  pathOptions={{ color: "#2196f3", weight: 6, opacity: 0.9 }}
                >
                  <Popup maxWidth={200}>
                    <div style={{ fontFamily: "system-ui, sans-serif", fontSize: "12px" }}>
                      <strong>🛣️ {match.road.name || match.road.highway}</strong>
                      <div style={{ color: "#666", marginTop: "4px" }}>OSM ID: {match.road.id}</div>
                      <div style={{ color: "#666" }}>Type: {match.road.highway}</div>
                    </div>
                  </Popup>
                </Polyline>
              )}

              {/* Crossing point markers for selected corridor */}
              {isSelected && (corridor.road_crossings || []).map((cr, ci) => (
                <CircleMarker
                  key={ci}
                  center={[cr.crossing_lat, cr.crossing_lon]}
                  radius={6}
                  pathOptions={{
                    color: "#1a237e",
                    fillColor: "#3f51b5",
                    fillOpacity: 0.8,
                    weight: 2,
                  }}
                >
                  <Popup maxWidth={220}>
                    <div style={{ fontFamily: "system-ui, sans-serif", fontSize: "12px", lineHeight: 1.5 }}>
                      <strong>Road Crossing</strong>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3px 8px", marginTop: "6px" }}>
                        <span style={{ color: "#888" }}>Road Type</span>
                        <span>{cr.road_type ?? "—"}</span>
                        <span style={{ color: "#888" }}>Events</span>
                        <span style={{ fontWeight: "600" }}>{cr.crossing_count}</span>
                        <span style={{ color: "#888" }}>Danger</span>
                        <span style={{ fontWeight: "700", color: cfg.color }}>{cr.danger_level}</span>
                        <span style={{ color: "#888" }}>Night</span>
                        <span>🌙 {cr.night_ratio != null ? (cr.night_ratio * 100).toFixed(0) + "%" : "—"}</span>
                        <span style={{ color: "#888" }}>Peak Time</span>
                        <span>{cr.peak_time_of_day ?? "—"}</span>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}

              {/* Corridor polyline */}
              <Polyline
                positions={positions}
                pathOptions={{
                  color:   isSelected ? "#1a237e" : cfg.color,
                  weight:  isSelected ? cfg.weight + 2 : cfg.weight,
                  opacity: isSelected ? 1 : cfg.opacity,
                  dashArray: isSelected ? null : "8,4",
                }}
                eventHandlers={{
                  click: () => setSelectedId(isSelected ? null : corridor.corridor_id),
                }}
              >
                <Popup maxWidth={300}>
                  <div style={{ fontFamily: "system-ui, sans-serif", lineHeight: 1.5 }}>
                    <div style={{
                      display: "flex", justifyContent: "space-between", alignItems: "center",
                      marginBottom: "8px", paddingBottom: "8px", borderBottom: "1px solid #eee"
                    }}>
                      <strong style={{ fontSize: "13px", color: "#222" }}>
                        Corridor {corridor.corridor_id}
                      </strong>
                      <span style={{
                        padding: "2px 8px", borderRadius: "10px", fontSize: "11px",
                        fontWeight: "700", color: "white", backgroundColor: cfg.color
                      }}>{dangerLvl}</span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 10px", fontSize: "12px" }}>
                      <span style={{ color: "#888" }}>Node Path</span>
                      <span>{corridor.from_node} → {corridor.to_node}</span>

                      <span style={{ color: "#888" }}>Road Crossings</span>
                      <span style={{ fontWeight: "600" }}>{(corridor.road_crossings || []).length}</span>

                      <span style={{ color: "#888" }}>Total Events</span>
                      <span style={{ fontWeight: "600" }}>
                        {(corridor.road_crossings || []).reduce((s, c) => s + (c.crossing_count || 0), 0)}
                      </span>

                      <span style={{ color: "#888" }}>Length</span>
                      <span>{corridor.distance_meters != null ? (corridor.distance_meters / 1000).toFixed(2) + " km" : "—"}</span>

                      <span style={{ color: "#888" }}>Safety Score</span>
                      <span style={{ fontWeight: "600", color: cfg.color }}>{corridor.safety_score?.toFixed(1) ?? "—"}</span>

                      {top && (
                        <>
                          <span style={{ color: "#888" }}>Highest Risk Road</span>
                          <span>{top.road_type ?? "—"}</span>
                          <span style={{ color: "#888" }}>Peak Time</span>
                          <span>{top.peak_time_of_day ?? "—"}</span>
                          <span style={{ color: "#888" }}>Night Activity</span>
                          <span>🌙 {top.night_ratio != null ? (top.night_ratio * 100).toFixed(0) + "%" : "—"}</span>
                        </>
                      )}
                    </div>
                    <div style={{ marginTop: "6px", paddingTop: "6px", borderTop: "1px solid #eee",
                                  fontSize: "10px", color: "#aaa" }}>
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
