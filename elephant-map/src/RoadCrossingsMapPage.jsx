import React, { useState, useEffect, useRef, useCallback } from "react";
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, useMap, useMapEvents } from "react-leaflet";
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
  High:   { color: "#d32f2f", fill: "#ef5350", radius: 9  },
  Medium: { color: "#e65100", fill: "#ff7043", radius: 7  },
  Low:    { color: "#f9a825", fill: "#ffca28", radius: 5  },
};

const ROAD_TYPES = ["All", "Primary", "Secondary", "Tertiary", "Track", "Unclassified", "Other"];
const SEASONS    = ["All", "Dry", "Wet", "Northeast Monsoon", "Southwest Monsoon"];
const TIMES      = ["All", "Morning", "Afternoon", "Evening", "Night"];

function FitBounds({ crossings }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    const valid = crossings.filter(c => c.crossing_lat != null && c.crossing_lon != null);
    if (!fitted.current && valid.length > 0) {
      const lats = valid.map(c => c.crossing_lat);
      const lngs = valid.map(c => c.crossing_lon);
      map.fitBounds([
        [Math.min(...lats), Math.min(...lngs)],
        [Math.max(...lats), Math.max(...lngs)],
      ], { padding: [40, 40] });
      fitted.current = true;
    }
  }, [crossings, map]);
  return null;
}

export default function RoadCrossingsMapPage() {
  const [crossings,     setCrossings]     = useState([]);
  const [loading,       setLoading]       = useState(true);
  const [error,         setError]         = useState(null);
  const [showPanel,     setShowPanel]     = useState(true);
  const [osmRoads,      setOsmRoads]      = useState([]);
  const [roadFetching,  setRoadFetching]  = useState(false);
  const [selectedId,    setSelectedId]    = useState(null);  // clicked crossing

  // Filters
  const [dangers,       setDangers]       = useState({ High: true, Medium: true, Low: true });
  const [roadType,      setRoadType]      = useState("All");
  const [peakSeason,    setPeakSeason]    = useState("All");
  const [peakTime,      setPeakTime]      = useState("All");
  const [minCrossings,  setMinCrossings]  = useState(1);
  const [minNight,      setMinNight]      = useState(0);

  useEffect(() => {
    axios.get(`${API}/road-crossings`)
      .then(r => { setCrossings(r.data); setLoading(false); })
      .catch(() => { setError("Cannot reach API — start the backend."); setLoading(false); });
  }, []);

  const filtered = crossings.filter(c => {
    if (!dangers[c.danger_level])                                          return false;
    if (roadType    !== "All" && c.road_type    !== roadType)              return false;
    if (peakSeason  !== "All" && c.peak_season  !== peakSeason)           return false;
    if (peakTime    !== "All" && c.peak_time_of_day !== peakTime)         return false;
    if (c.crossing_count < minCrossings)                                   return false;
    if ((c.night_ratio * 100) < minNight)                                  return false;
    return true;
  });

  const counts = { High: 0, Medium: 0, Low: 0 };
  filtered.forEach(c => counts[c.danger_level]++);

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
        <span style={{ fontWeight: "700", color: "#333" }}>Showing {filtered.length.toLocaleString()} crossings</span>
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
            <span style={{ fontWeight: "700", fontSize: "13px" }}>🚗 Road Crossings Map</span>
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
        border: "1px solid rgba(0,0,0,0.08)", minWidth: "160px"
      }}>
        <div style={{ fontWeight: "700", fontSize: "12px", color: "#333", marginBottom: "8px" }}>
          Danger Level
        </div>
        {Object.entries(DANGER_CONFIG).map(([lvl, cfg]) => (
          <div key={lvl} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "5px" }}>
            <svg width={cfg.radius * 2 + 2} height={cfg.radius * 2 + 2} style={{ flexShrink: 0 }}>
              <circle cx={cfg.radius + 1} cy={cfg.radius + 1} r={cfg.radius}
                fill={cfg.fill} stroke={cfg.color} strokeWidth="1.5" />
            </svg>
            <span style={{ fontSize: "12px", color: "#444" }}>{lvl}</span>
          </div>
        ))}
        <div style={{ marginTop: "8px", paddingTop: "8px", borderTop: "1px solid #eee",
                      fontSize: "10px", color: "#999" }}>
          Click a crossing to show nearest road
        </div>
        <div style={{ marginTop: "6px", display: "flex", alignItems: "center", gap: "6px" }}>
          <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="#2196f3" strokeWidth="3" /></svg>
          <span style={{ fontSize: "11px", color: "#444" }}>Matched Road</span>
        </div>
        <div style={{ marginTop: "4px", display: "flex", alignItems: "center", gap: "6px" }}>
          <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="#888" strokeWidth="1.5" strokeDasharray="3,3" /></svg>
          <span style={{ fontSize: "11px", color: "#444" }}>Connector</span>
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
        <FitBounds crossings={filtered} />
        <RoadFetcher onRoads={setOsmRoads} onFetching={setRoadFetching} />

        {filtered.map((c, i) => {
          const cfg  = DANGER_CONFIG[c.danger_level] || DANGER_CONFIG.Low;
          const lat  = c.crossing_lat;
          const lng  = c.crossing_lon;
          if (!lat || !lng) return null;

          const crossingKey = c.crossing_id ?? i;
          const isSelected = selectedId === crossingKey;

          // Find nearest road only for selected crossing
          const match = isSelected && osmRoads.length > 0
            ? findNearestRoad(lat, lng, osmRoads)
            : null;

          return (
            <React.Fragment key={crossingKey}>
              {/* Highlighted road — only for selected */}
              {isSelected && match && (
                <Polyline
                  positions={match.road.coords}
                  pathOptions={{
                    color: "#2196f3",
                    weight: 5,
                    opacity: 0.85,
                  }}
                >
                  <Popup maxWidth={200}>
                    <div style={{ fontFamily: "system-ui, sans-serif", fontSize: "12px" }}>
                      <strong>🛣️ {match.road.name || match.road.highway}</strong>
                      <div style={{ color: "#666", marginTop: "4px" }}>OSM ID: {match.road.id}</div>
                    </div>
                  </Popup>
                </Polyline>
              )}

              {/* Dotted connector line from node to road — only for selected */}
              {isSelected && match && match.closestPoint && (
                <Polyline
                  positions={[[lat, lng], match.closestPoint]}
                  pathOptions={{
                    color: "#555",
                    weight: 2,
                    dashArray: "5,5",
                    opacity: 0.7,
                  }}
                />
              )}

              {/* Crossing point marker */}
              <CircleMarker
                center={[lat, lng]}
                radius={isSelected ? cfg.radius + 3 : cfg.radius}
                pathOptions={{
                  color:       isSelected ? "#1a237e" : cfg.color,
                  fillColor:   cfg.fill,
                  fillOpacity: isSelected ? 1 : 0.85,
                  weight:      isSelected ? 3 : 1.5,
                }}
                eventHandlers={{
                  click: () => setSelectedId(isSelected ? null : crossingKey),
                }}
              >
              <Popup maxWidth={280}>
                <div style={{ fontFamily: "system-ui, sans-serif", lineHeight: 1.5 }}>
                  {/* Title */}
                  <div style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    marginBottom: "8px", paddingBottom: "8px", borderBottom: "1px solid #eee"
                  }}>
                    <strong style={{ fontSize: "13px", color: "#222" }}>
                      Road Crossing {c.crossing_id != null
                        ? `#${c.crossing_id}`
                        : `@ ${lat.toFixed(4)}, ${lng.toFixed(4)}`}
                    </strong>
                    <span style={{
                      padding: "2px 8px", borderRadius: "10px", fontSize: "11px",
                      fontWeight: "700", color: "white",
                      backgroundColor: cfg.color
                    }}>{c.danger_level}</span>
                  </div>

                  {/* Metrics grid */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 10px", fontSize: "12px" }}>
                    <span style={{ color: "#888" }}>Danger Score</span>
                    <span style={{ fontWeight: "700", color: cfg.color }}>{c.danger_score?.toFixed(1) ?? "—"}</span>

                    <span style={{ color: "#888" }}>Corridor</span>
                    <span>{c.corridor_id ?? "—"}</span>

                    <span style={{ color: "#888" }}>Crossing Events</span>
                    <span style={{ fontWeight: "600" }}>{c.crossing_count?.toLocaleString() ?? "—"}</span>

                    <span style={{ color: "#888" }}>Elephants</span>
                    <span>{c.elephant_count ?? "—"}</span>

                    <span style={{ color: "#888" }}>Night Activity</span>
                    <span style={{ color: "#1a237e", fontWeight: "600" }}>
                      🌙 {c.night_ratio != null ? (c.night_ratio * 100).toFixed(0) + "%" : "—"}
                    </span>

                    <span style={{ color: "#888" }}>Peak Hours</span>
                    <span>{Array.isArray(c.peak_hours) ? c.peak_hours.join(", ") + "h" : "—"}</span>

                    <span style={{ color: "#888" }}>Peak Season</span>
                    <span>{c.peak_season ?? "—"}</span>

                    <span style={{ color: "#888" }}>Peak Time</span>
                    <span>{c.peak_time_of_day ?? "—"}</span>

                    <span style={{ color: "#888" }}>Road Type</span>
                    <span>{c.road_type ?? "—"}</span>

                    <span style={{ color: "#888" }}>Traffic Exposure</span>
                    <span>{c.traffic_exposure != null ? (c.traffic_exposure * 100).toFixed(0) + "%" : "—"}</span>
                  </div>

                  {/* Coordinates */}
                  <div style={{ marginTop: "8px", paddingTop: "6px", borderTop: "1px solid #eee",
                                fontSize: "10px", color: "#aaa" }}>
                    {lat.toFixed(5)}, {lng.toFixed(5)}
                  </div>
                </div>
              </Popup>
              </CircleMarker>
            </React.Fragment>
          );
        })}
      </MapContainer>
    </div>
  );
}
