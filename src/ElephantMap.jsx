import { useState, useEffect } from "react";
import { MapContainer, TileLayer, Circle, Marker, Popup, Polyline, Polygon, useMapEvents } from "react-leaflet";
import { Link } from "react-router-dom";
import axios from "axios";
import { DISHAN_API } from "./apiConfig";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { RefreshCw } from "lucide-react";

// Function to calculate rectangle around corridor based on human distance
const calculateCorridorRectangle = (corridor) => {
  const path = corridor.path;
  if (path.length < 2) return null;
  const start = path[0];
  const end = path[path.length - 1];
  const dx = end.lon - start.lon;
  const dy = end.lat - start.lat;
  const length = Math.sqrt(dx * dx + dy * dy);
  if (length === 0) return null;
  const dirX = dx / length;
  const dirY = dy / length;
  const perpX = -dirY;
  const perpY = dirX;
  const avgLat = (start.lat + end.lat) / 2;
  const metersPerDegreeLat = 111320;
  const metersPerDegreeLon = 111320 * Math.cos(avgLat * Math.PI / 180);
  const offsetLat = corridor.avg_human_distance / metersPerDegreeLat;
  const offsetLon = corridor.avg_human_distance / metersPerDegreeLon;
  const corner1 = [start.lat + perpY * offsetLat, start.lon + perpX * offsetLon];
  const corner2 = [start.lat - perpY * offsetLat, start.lon - perpX * offsetLon];
  const corner3 = [end.lat - perpY * offsetLat, end.lon - perpX * offsetLon];
  const corner4 = [end.lat + perpY * offsetLat, end.lon + perpX * offsetLon];
  return [corner1, corner4, corner3, corner2];
};

const getDistrict = (lat, lon) => {
  if (lat >= 6.0 && lat <= 6.5 && lon >= 80.0 && lon <= 81.5) {
    if (lon < 80.5) return "Galle";
    if (lon < 81.0) return "Matara";
    return "Hambantota";
  }
  if (lat >= 6.5 && lat <= 7.5 && lon >= 81.0 && lon <= 81.8) {
    if (lat < 6.8) return "Monaragala";
    return "Badulla";
  }
  if (lat >= 6.5 && lat <= 7.5 && lon >= 80.2 && lon <= 81.0) {
    if (lat < 7.0) return "Ratnapura";
    return "Kegalle";
  }
  if (lat >= 7.0 && lat <= 7.5 && lon >= 80.5 && lon <= 81.3) {
    if (lon < 80.8) return "Kandy";
    if (lat < 7.3) return "Matale";
    return "Nuwara Eliya";
  }
  if (lat >= 7.5 && lat <= 8.5 && lon >= 80.0 && lon <= 81.3) {
    if (lon < 80.5) return "Anuradhapura";
    return "Polonnaruwa";
  }
  if (lat >= 7.0 && lon >= 81.0) {
    if (lat < 7.5) return "Ampara";
    if (lat < 8.5) return "Batticaloa";
    return "Trincomalee";
  }
  if (lat >= 6.5 && lat <= 7.5 && lon >= 79.5 && lon <= 80.5) {
    if (lat > 7.0 && lon > 79.8 && lon < 80.2) return "Colombo";
    if (lat < 7.0) return "Kalutara";
    return "Gampaha";
  }
  if (lat >= 7.0 && lat <= 8.5 && lon >= 79.5 && lon <= 80.5) {
    if (lat < 7.8) return "Kurunegala";
    return "Puttalam";
  }
  return "Other";
};

const G = {
  panelBg: "rgba(10,14,20,0.95)",
  card: "rgba(255,255,255,0.07)",
  border: "rgba(255,255,255,0.12)",
  inputBg: "rgba(255,255,255,0.08)",
  textPrimary: "#f9fafb",
  textSecondary: "rgba(255,255,255,0.65)",
  textMuted: "rgba(255,255,255,0.38)",
  emerald: "#10b981",
  emeraldLight: "#34d399",
};

export default function ElephantMap() {
  const [nodes, setNodes] = useState([]);
  const [corridors, setCorridors] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [zoom, setZoom] = useState(8);
  const [selectedDistrict, setSelectedDistrict] = useState("All");
  const [districts, setDistricts] = useState([]);
  const [dangerLevel, setDangerLevel] = useState("All");
  const [elephantCountFilter, setElephantCountFilter] = useState("All");
  const [timeFilter, setTimeFilter] = useState("All");
  const [showCorridors, setShowCorridors] = useState(true);
  const [selectedCorridor, setSelectedCorridor] = useState(null);
  const [humanDistanceRect, setHumanDistanceRect] = useState(null);

  function ZoomTracker() {
    useMapEvents({ zoomend: (e) => setZoom(e.target.getZoom()) });
    return null;
  }

  const elephantIcon = L.divIcon({
    html: '<div style="font-size: 16px; text-align: center;">🐘</div>',
    className: 'custom-elephant-icon',
    iconSize: [20, 20], iconAnchor: [10, 10], popupAnchor: [0, -10]
  });

  useEffect(() => {
    setLoading(true);
    axios.get(`${DISHAN_API}/nodes`)
      .then((response) => {
        const nodesWithDistricts = response.data.map(node => ({
          ...node, district: getDistrict(node.center_lat, node.center_lon)
        }));
        setNodes(nodesWithDistricts);
        const uniqueDistricts = [...new Set(nodesWithDistricts.map(n => n.district))].sort();
        setDistricts(uniqueDistricts);
        setError(null);
        setLoading(false);
      })
      .catch((err) => { setError(err.message); setLoading(false); });

    axios.get(`${DISHAN_API}/corridors`)
      .then((response) => setCorridors(response.data))
      .catch(() => {});
  }, []);

  const getDangerLevel = (score) => {
    if (score < 30) return "High";
    if (score < 60) return "Medium";
    return "Low";
  };

  const filteredNodes = nodes.filter(n => {
    if (selectedDistrict !== "All" && n.district !== selectedDistrict) return false;
    if (dangerLevel !== "All" && getDangerLevel(n.safety_score) !== dangerLevel) return false;
    if (elephantCountFilter !== "All") {
      if (elephantCountFilter === "1" && n.elephant_count !== 1) return false;
      if (elephantCountFilter === "2" && n.elephant_count !== 2) return false;
      if (elephantCountFilter === "3+" && n.elephant_count < 3) return false;
    }
    if (timeFilter !== "All" && !n.active_hours.includes(parseInt(timeFilter))) return false;
    return true;
  });

  const getColor = (safety_score) => {
    if (safety_score < 30) return "#ef4444";
    if (safety_score < 60) return "#f59e0b";
    if (safety_score < 80) return "#fbbf24";
    return "#10b981";
  };

  const getOpacity = (elephant_count) => Math.min(0.2 + (elephant_count * 0.1), 0.7);
  const getCorridorWidth = (usage_count) => Math.min(3 + usage_count, 8);
  const getCorridorOpacity = (safety_score) => {
    if (safety_score < 30) return 0.85;
    if (safety_score < 60) return 0.75;
    return 0.65;
  };

  const handleCorridorClick = (corridor) => {
    if (selectedCorridor === corridor.corridor_id) {
      setSelectedCorridor(null);
      setHumanDistanceRect(null);
    } else {
      setSelectedCorridor(corridor.corridor_id);
      const rect = calculateCorridorRectangle(corridor);
      setHumanDistanceRect({ rect, corridor });
    }
  };

  const selectStyle = {
    width: "100%", padding: "8px 10px", borderRadius: "8px",
    border: `1px solid ${G.border}`, background: G.inputBg,
    color: G.textPrimary, fontSize: "12px", cursor: "pointer", outline: "none",
  };

  const labelStyle = {
    fontSize: "11px", color: G.textMuted, fontWeight: "700",
    textTransform: "uppercase", letterSpacing: "0.5px", display: "block", marginBottom: "6px",
  };

  return (
    <div>
      {error && (
        <div style={{
          position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)",
          zIndex: 2000, background: "rgba(239,68,68,0.15)", backdropFilter: "blur(20px)",
          border: "1px solid rgba(239,68,68,0.4)", color: "#f87171",
          padding: "24px 28px", borderRadius: "16px", boxShadow: "0 8px 32px rgba(0,0,0,0.5)", maxWidth: "400px",
          fontFamily: "'Inter', sans-serif",
        }}>
          <h3 style={{ margin: "0 0 10px 0", color: "#ef4444" }}>API Connection Error</h3>
          <p style={{ margin: 0, fontSize: "14px" }}>{error}</p>
          <p style={{ margin: "10px 0 0 0", fontSize: "12px", color: G.textMuted }}>
            Make sure the API server is running on {DISHAN_API}
          </p>
        </div>
      )}

      {loading && !error && (
        <div style={{
          position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)",
          zIndex: 2000, background: G.panelBg, backdropFilter: "blur(20px)",
          border: `1px solid ${G.border}`, color: G.textPrimary,
          padding: "24px 32px", borderRadius: "16px", boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
          display: "flex", alignItems: "center", gap: "12px",
          fontFamily: "'Inter', sans-serif",
        }}>
          <RefreshCw size={18} style={{ animation: "spin 1s linear infinite", color: G.emerald }} />
          <span>Loading elephant data…</span>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      <MapContainer
        center={[7.8731, 80.7718]}
        zoom={8}
        style={{ height: "calc(100vh - 60px)", width: "100%" }}
        maxBounds={[[5.5, 79.0], [10.0, 82.5]]}
        minZoom={7}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <ZoomTracker />

        {/* Filter Panel */}
        <div style={{
          position: "absolute", top: "10px", right: "10px", zIndex: 1000,
          background: G.panelBg, backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
          padding: "18px", borderRadius: "16px",
          boxShadow: "0 8px 32px rgba(0,0,0,0.6)", border: `1px solid ${G.border}`,
          maxWidth: "270px", maxHeight: "calc(100vh - 80px)", overflowY: "auto",
          fontFamily: "'Inter', -apple-system, sans-serif",
        }}>
          <h3 style={{ margin: "0 0 14px 0", fontSize: "14px", fontWeight: "700", color: G.emeraldLight, display: "flex", alignItems: "center", gap: "6px" }}>
            🐘 Map Filters
          </h3>

          {/* Results Count */}
          <div style={{ marginBottom: "14px", padding: "10px", background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)", borderRadius: "10px", textAlign: "center" }}>
            <strong style={{ color: G.emeraldLight, fontSize: "20px", fontWeight: "800" }}>{filteredNodes.length}</strong>
            <span style={{ color: G.textMuted, fontSize: "12px", marginLeft: "6px" }}>location{filteredNodes.length !== 1 ? 's' : ''}</span>
          </div>

          {/* Show Corridors Toggle */}
          <div style={{ marginBottom: "14px", padding: "8px 10px", background: G.card, border: `1px solid ${G.border}`, borderRadius: "8px" }}>
            <label style={{ fontSize: "13px", color: G.textSecondary, display: "flex", alignItems: "center", cursor: "pointer", gap: "8px" }}>
              <input
                type="checkbox"
                checked={showCorridors}
                onChange={(e) => setShowCorridors(e.target.checked)}
                style={{ cursor: "pointer", accentColor: G.emerald }}
              />
              Show Elephant Corridors
            </label>
          </div>

          {/* Danger Level Filter */}
          <div style={{ marginBottom: "14px" }}>
            <span style={labelStyle}>Danger Level</span>
            <select value={dangerLevel} onChange={(e) => setDangerLevel(e.target.value)} style={selectStyle}>
              <option value="All">All Levels</option>
              <option value="High">🔴 High Danger (&lt;30)</option>
              <option value="Medium">🟠 Medium Danger (30-60)</option>
              <option value="Low">🟢 Low Danger (&gt;60)</option>
            </select>
          </div>

          {/* Elephant Count Filter */}
          <div style={{ marginBottom: "14px" }}>
            <span style={labelStyle}>Elephant Count</span>
            <select value={elephantCountFilter} onChange={(e) => setElephantCountFilter(e.target.value)} style={selectStyle}>
              <option value="All">All Counts</option>
              <option value="1">1 Elephant</option>
              <option value="2">2 Elephants</option>
              <option value="3+">3+ Elephants</option>
            </select>
          </div>

          {/* District Filter */}
          <div style={{ marginBottom: "14px" }}>
            <span style={labelStyle}>Region / District</span>
            <select value={selectedDistrict} onChange={(e) => setSelectedDistrict(e.target.value)} style={selectStyle}>
              <option value="All">All Districts</option>
              {districts.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          {/* Time Period Filter */}
          <div style={{ marginBottom: "14px" }}>
            <span style={labelStyle}>Time Period (Active Hours)</span>
            <select value={timeFilter} onChange={(e) => setTimeFilter(e.target.value)} style={selectStyle}>
              <option value="All">All Hours</option>
              <option value="0">00:00 - 01:00 (Midnight)</option>
              <option value="4">04:00 - 05:00 (Dawn)</option>
              <option value="8">08:00 - 09:00 (Morning)</option>
              <option value="12">12:00 - 13:00 (Noon)</option>
              <option value="16">16:00 - 17:00 (Afternoon)</option>
              <option value="20">20:00 - 21:00 (Evening)</option>
            </select>
          </div>

          {/* Reset Button */}
          <button
            onClick={() => {
              setDangerLevel("All"); setElephantCountFilter("All");
              setSelectedDistrict("All"); setTimeFilter("All");
              setSelectedCorridor(null); setHumanDistanceRect(null);
            }}
            style={{
              width: "100%", padding: "10px", fontSize: "13px", fontWeight: "600",
              background: "rgba(239,68,68,0.15)", color: "#f87171",
              border: "1px solid rgba(239,68,68,0.3)", borderRadius: "10px", cursor: "pointer",
              marginBottom: selectedCorridor ? "10px" : "0", transition: "all 0.2s",
            }}
            onMouseEnter={e => e.currentTarget.style.background = "rgba(239,68,68,0.25)"}
            onMouseLeave={e => e.currentTarget.style.background = "rgba(239,68,68,0.15)"}
          >
            Reset All Filters
          </button>

          {selectedCorridor && (
            <button
              onClick={() => { setSelectedCorridor(null); setHumanDistanceRect(null); }}
              style={{
                width: "100%", padding: "10px", fontSize: "13px", fontWeight: "600",
                background: "rgba(96,165,250,0.15)", color: "#60a5fa",
                border: "1px solid rgba(96,165,250,0.3)", borderRadius: "10px", cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              Hide Human Distance Zone
            </button>
          )}
        </div>

        {/* Legend */}
        <div style={{
          position: "absolute", bottom: "20px", left: "10px", zIndex: 1000,
          background: G.panelBg, backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
          padding: "14px 16px", borderRadius: "14px",
          boxShadow: "0 8px 32px rgba(0,0,0,0.5)", border: `1px solid ${G.border}`,
          fontFamily: "'Inter', -apple-system, sans-serif",
        }}>
          <h4 style={{ margin: "0 0 12px 0", fontSize: "12px", fontWeight: "700", color: G.emeraldLight, textTransform: "uppercase", letterSpacing: "0.5px" }}>Safety Levels</h4>
          {[
            { color: "#ef4444", label: "High Risk (<30)" },
            { color: "#f59e0b", label: "Medium (30-60)" },
            { color: "#fbbf24", label: "Moderate (60-80)" },
            { color: "#10b981", label: "Low Risk (>80)" },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", marginBottom: "7px" }}>
              <div style={{ width: "16px", height: "16px", background: color, marginRight: "10px", borderRadius: "50%", boxShadow: `0 0 8px ${color}60`, flexShrink: 0 }} />
              <span style={{ fontSize: "12px", color: G.textSecondary }}>{label}</span>
            </div>
          ))}
          <div style={{ borderTop: `1px solid ${G.border}`, paddingTop: "10px", marginTop: "6px" }}>
            <div style={{ fontSize: "11px", color: G.textMuted, marginBottom: "5px" }}>🐘 Hotspot Zones</div>
            <div style={{ fontSize: "11px", color: G.textMuted, display: "flex", alignItems: "center", marginBottom: "5px" }}>
              <div style={{ width: "30px", height: "3px", background: "linear-gradient(90deg, #ef4444 0%, #ef4444 50%, transparent 50%, transparent 100%)", backgroundSize: "10px 3px", marginRight: "8px" }} />
              <span>Corridors</span>
            </div>
            <div style={{ fontSize: "11px", color: G.textMuted, display: "flex", alignItems: "center" }}>
              <div style={{ width: "20px", height: "12px", background: "rgba(96,165,250,0.2)", marginRight: "8px", border: "1px dashed #60a5fa" }} />
              <span>Human Distance (click corridor)</span>
            </div>
          </div>
        </div>

        {/* Corridors */}
        {showCorridors && corridors.map((corridor, i) => {
          const pathCoords = corridor.path.map(p => [p.lat, p.lon]);
          const corridorColor = getColor(corridor.safety_score);
          const corridorWidth = getCorridorWidth(corridor.usage_count);
          const corridorOpacity = getCorridorOpacity(corridor.safety_score);
          const isHighDanger = corridor.safety_score < 30;
          return (
            <>
              {isHighDanger && (
                <Polyline key={`glow-${i}`} positions={pathCoords}
                  pathOptions={{ color: corridorColor, weight: corridorWidth + 4, opacity: 0.2, dashArray: "10, 10", lineCap: "round", lineJoin: "round" }}
                />
              )}
              <Polyline key={`corridor-${i}`} positions={pathCoords}
                pathOptions={{ color: corridorColor, weight: corridorWidth, opacity: corridorOpacity, dashArray: "10, 8", lineCap: "round", lineJoin: "round" }}
                eventHandlers={{
                  click: () => handleCorridorClick(corridor),
                  mouseover: (e) => e.target.setStyle({ weight: corridorWidth + 2, opacity: 1 }),
                  mouseout: (e) => e.target.setStyle({ weight: corridorWidth, opacity: corridorOpacity }),
                }}
              >
                <Popup>
                  <div style={{ minWidth: "220px", fontFamily: "'Inter', sans-serif" }}>
                    <div style={{ borderBottom: `2px solid ${corridorColor}`, paddingBottom: "8px", marginBottom: "8px", fontWeight: "700", fontSize: "14px", color: "#222" }}>
                      🐘 Elephant Corridor
                    </div>
                    <div style={{ fontSize: "12px", lineHeight: "1.7", color: "#444" }}>
                      <div><strong>ID:</strong> {corridor.corridor_id}</div>
                      <div><strong>Route:</strong> Node {corridor.from_node} → {corridor.to_node}</div>
                      <div><strong>Distance:</strong> {(corridor.distance_meters / 1000).toFixed(2)} km</div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span><strong>Usage:</strong> {corridor.usage_count}×</span>
                        <span><strong>Crossings:</strong> {corridor.crossing_count}</span>
                      </div>
                      <div><strong>Active Hours:</strong> {corridor.active_hours.map(h => h + ':00').join(", ")}</div>
                      <div><strong>Human Distance:</strong> {(corridor.avg_human_distance / 1000).toFixed(2)} km</div>
                      <div style={{ marginTop: "8px", padding: "6px", background: corridorColor + "20", borderRadius: "6px", textAlign: "center", fontWeight: "600" }}>
                        Safety Score: {corridor.safety_score.toFixed(1)}
                      </div>
                    </div>
                  </div>
                </Popup>
              </Polyline>
            </>
          );
        })}

        {/* Human Distance Rectangle */}
        {humanDistanceRect && humanDistanceRect.rect && (
          <Polygon positions={humanDistanceRect.rect}
            pathOptions={{ color: "#60a5fa", weight: 2, fillColor: "#60a5fa", fillOpacity: 0.12, dashArray: "5, 5" }}
          >
            <Popup>
              <div style={{ minWidth: "200px", fontFamily: "'Inter', sans-serif" }}>
                <div style={{ borderBottom: "2px solid #60a5fa", paddingBottom: "8px", marginBottom: "8px", fontWeight: "700", fontSize: "14px", color: "#222" }}>
                  📏 Human Distance Zone
                </div>
                <div style={{ fontSize: "12px", lineHeight: "1.7", color: "#444" }}>
                  <div><strong>Corridor:</strong> {humanDistanceRect.corridor.corridor_id}</div>
                  <div><strong>Avg Human Distance:</strong> {(humanDistanceRect.corridor.avg_human_distance / 1000).toFixed(2)} km</div>
                  <div><strong>Distance (meters):</strong> {humanDistanceRect.corridor.avg_human_distance.toFixed(0)} m</div>
                  <div style={{ marginTop: "8px", padding: "6px", background: "#60a5fa20", borderRadius: "4px", textAlign: "center", fontSize: "11px" }}>
                    Rectangle width represents the average distance to human settlements from this corridor
                  </div>
                </div>
              </div>
            </Popup>
          </Polygon>
        )}

        {/* Node circles & markers */}
        {filteredNodes.map((n, i) => {
          const nodeColor = getColor(n.safety_score);
          return (
            <div key={i}>
              <Circle center={[n.center_lat, n.center_lon]} radius={n.radius_meters + 50}
                pathOptions={{ color: nodeColor, weight: 0, fillColor: nodeColor, fillOpacity: 0.08 }}
              />
              <Circle center={[n.center_lat, n.center_lon]} radius={n.radius_meters}
                pathOptions={{ color: nodeColor, weight: 2, fillColor: nodeColor, fillOpacity: getOpacity(n.elephant_count) }}
              />
              {zoom >= 10 && (
                <Marker position={[n.center_lat, n.center_lon]} icon={elephantIcon}>
                  <Popup>
                    <div style={{ minWidth: "200px", fontFamily: "'Inter', sans-serif" }}>
                      <div style={{ borderBottom: `2px solid ${nodeColor}`, paddingBottom: "8px", marginBottom: "8px", fontWeight: "700", fontSize: "14px", color: "#222" }}>
                        🐘 Elephant Hotspot
                      </div>
                      <div style={{ fontSize: "12px", lineHeight: "1.7", color: "#444" }}>
                        <div><strong>District:</strong> {n.district}</div>
                        <div><strong>Node ID:</strong> {n.node_id}</div>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span><strong>Elephants:</strong> {n.elephant_count}</span>
                          <span><strong>Sightings:</strong> {n.sighting_count}</span>
                        </div>
                        <div><strong>Active Hours:</strong> {n.active_hours.map(h => h + ':00').join(", ")}</div>
                        <div><strong>NDVI:</strong> {n.avg_ndvi.toFixed(3)}</div>
                        <div><strong>Human Distance:</strong> {(n.avg_human_distance / 1000).toFixed(2)} km</div>
                        <div style={{ marginTop: "8px", padding: "6px", background: nodeColor + "20", borderRadius: "4px", textAlign: "center", fontWeight: "600" }}>
                          Safety Score: {n.safety_score.toFixed(1)}
                        </div>
                      </div>
                    </div>
                  </Popup>
                </Marker>
              )}
            </div>
          );
        })}
      </MapContainer>
    </div>
  );
}
