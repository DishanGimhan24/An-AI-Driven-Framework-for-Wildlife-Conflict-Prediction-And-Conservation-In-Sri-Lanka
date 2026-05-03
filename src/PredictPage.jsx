import { useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, Circle, Polyline, useMapEvents } from "react-leaflet";
import axios from "axios";
import { DISHAN_API } from "./apiConfig";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapPin, Calendar, Clock, Navigation } from "lucide-react";

const pinIcon = L.divIcon({
  html: '<div style="font-size: 28px; text-shadow: 2px 2px 6px rgba(0,0,0,0.5);">📍</div>',
  className: 'custom-pin-icon', iconSize: [28, 28], iconAnchor: [14, 28], popupAnchor: [0, -28]
});
const elephantIcon = L.divIcon({
  html: '<div style="font-size: 24px;">🐘</div>',
  className: 'custom-elephant-icon', iconSize: [24, 24], iconAnchor: [12, 12], popupAnchor: [0, -12]
});

function LocationPicker({ onLocationSelect }) {
  useMapEvents({ click: (e) => onLocationSelect(e.latlng) });
  return null;
}

const G = {
  panelBg: "#111827",
  card: "rgba(255,255,255,0.06)",
  border: "rgba(255,255,255,0.10)",
  inputBg: "rgba(255,255,255,0.07)",
  textPrimary: "#f9fafb",
  textSecondary: "rgba(255,255,255,0.65)",
  textMuted: "rgba(255,255,255,0.38)",
  emerald: "#10b981",
  emeraldLight: "#34d399",
};

const getRiskColor = (lvl) => ({ High: "#ef4444", Medium: "#f59e0b", Low: "#10b981" }[lvl] ?? "#6b7280");
const getRiskBg    = (lvl) => ({ High: "rgba(239,68,68,0.15)", Medium: "rgba(245,158,11,0.15)", Low: "rgba(16,185,129,0.15)" }[lvl] ?? "rgba(107,114,128,0.15)");
const getRiskBorder = (lvl) => ({ High: "rgba(239,68,68,0.4)", Medium: "rgba(245,158,11,0.4)", Low: "rgba(16,185,129,0.4)" }[lvl] ?? "rgba(107,114,128,0.4)");

export default function PredictPage() {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [hour, setHour] = useState(new Date().getHours());
  const [showNavigationModal, setShowNavigationModal] = useState(false);
  const [navigationTarget, setNavigationTarget] = useState(null);

  const handleLocationSelect = async (latlng) => {
    setSelectedLocation(latlng);
    setLoading(true);
    setError(null);
    try {
      const dateTime = new Date(selectedDate);
      dateTime.setHours(hour, 0, 0, 0);
      const response = await axios.post(`${DISHAN_API}/predict_risk`, {
        latitude: latlng.lat, longitude: latlng.lng, datetime: dateTime.toISOString()
      });
      const result = response.data;
      setPrediction(result);
      if (result.distance_to_node > 25000) {
        setNavigationTarget({
          distanceKm: (result.distance_to_node / 1000).toFixed(1),
          targetLat: result.nearest_node.center_lat,
          targetLon: result.nearest_node.center_lon,
          nodeId: result.nearest_node.node_id
        });
        setShowNavigationModal(true);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to fetch prediction");
    } finally {
      setLoading(false);
    }
  };

  const handleNavigateToZone = () => {
    if (navigationTarget) {
      const loc = { lat: navigationTarget.targetLat, lng: navigationTarget.targetLon };
      setSelectedLocation(loc);
      setShowNavigationModal(false);
      handleLocationSelect(loc);
    }
  };

  const inputStyle = {
    width: "100%", padding: "10px 14px", fontSize: "14px", borderRadius: "10px",
    border: `1px solid ${G.border}`, background: G.inputBg, color: G.textPrimary,
    cursor: "pointer", outline: "none", boxSizing: "border-box",
  };

  return (
    <div style={{ height: "calc(100vh - 60px)", display: "flex", flexDirection: "column", fontFamily: "'Inter', -apple-system, sans-serif" }}>
      <style>{`
        @keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.15); opacity: 0.7; } }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes slideIn { from { opacity: 0; transform: translateY(-16px) scale(0.97); } to { opacity: 1; transform: translateY(0) scale(1); } }
      `}</style>

      <div style={{ flex: 1, display: "flex", position: "relative" }}>
        {/* Map */}
        <div style={{ flex: 1 }}>
          <MapContainer center={[7.8731, 80.7718]} zoom={8}
            style={{ height: "100%", width: "100%" }}
            maxBounds={[[5.8, 79.4], [9.9, 82.0]]} minZoom={7} maxZoom={15} maxBoundsViscosity={1.0}
          >
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            <LocationPicker onLocationSelect={handleLocationSelect} />

            {selectedLocation && (
              <Marker position={[selectedLocation.lat, selectedLocation.lng]} icon={pinIcon}>
                <Popup>
                  <div style={{ textAlign: "center", fontFamily: "'Inter', sans-serif" }}>
                    <strong>Selected Location</strong><br />
                    Lat: {selectedLocation.lat.toFixed(6)}<br />
                    Lon: {selectedLocation.lng.toFixed(6)}
                  </div>
                </Popup>
              </Marker>
            )}

            {prediction?.nearest_node && (
              <>
                <Circle center={[prediction.nearest_node.center_lat, prediction.nearest_node.center_lon]}
                  radius={prediction.nearest_node.radius_meters}
                  pathOptions={{ color: "#f59e0b", weight: 2, fillColor: "#f59e0b", fillOpacity: 0.2 }}
                />
                <Marker position={[prediction.nearest_node.center_lat, prediction.nearest_node.center_lon]} icon={elephantIcon}>
                  <Popup>
                    <div style={{ fontFamily: "'Inter', sans-serif" }}>
                      <strong>🐘 Nearest Hotspot</strong><br />
                      Node ID: {prediction.nearest_node.node_id}<br />
                      Elephants: {prediction.nearest_node.elephant_count}<br />
                      Distance: {(prediction.distance_to_node / 1000).toFixed(2)} km
                    </div>
                  </Popup>
                </Marker>
              </>
            )}

            {prediction?.nearest_corridor && (
              <Polyline positions={prediction.nearest_corridor.path.map(p => [p.lat, p.lon])}
                pathOptions={{ color: "#a78bfa", weight: 4, opacity: 0.7, dashArray: "10, 8" }}
              >
                <Popup>
                  <div style={{ fontFamily: "'Inter', sans-serif" }}>
                    <strong>🛤️ Nearest Corridor</strong><br />
                    ID: {prediction.nearest_corridor.corridor_id}<br />
                    Distance: {(prediction.distance_to_corridor / 1000).toFixed(2)} km
                  </div>
                </Popup>
              </Polyline>
            )}

            {selectedLocation && prediction?.nearest_node && (
              <Polyline
                positions={[[selectedLocation.lat, selectedLocation.lng], [prediction.nearest_node.center_lat, prediction.nearest_node.center_lon]]}
                pathOptions={{ color: "#6b7280", weight: 2, opacity: 0.5, dashArray: "5, 10" }}
              />
            )}
          </MapContainer>
        </div>

        {/* Side Panel */}
        <div style={{
          width: "400px", background: G.panelBg, overflowY: "auto",
          borderLeft: `1px solid ${G.border}`, display: "flex", flexDirection: "column",
        }}>
          {/* Date & Time Selector */}
          <div style={{ padding: "18px", borderBottom: `1px solid ${G.border}`, background: "rgba(255,255,255,0.03)" }}>
            <label style={{ fontWeight: "700", fontSize: "13px", display: "flex", alignItems: "center", gap: "6px", marginBottom: "10px", color: G.textSecondary, textTransform: "uppercase", letterSpacing: "0.5px" }}>
              <Calendar size={14} color={G.emerald} /> Select Date
            </label>
            <input type="date" value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)}
              style={{ ...inputStyle, marginBottom: "14px" }}
            />
            <label style={{ fontWeight: "700", fontSize: "13px", display: "flex", alignItems: "center", gap: "6px", marginBottom: "10px", color: G.textSecondary, textTransform: "uppercase", letterSpacing: "0.5px" }}>
              <Clock size={14} color={G.emerald} /> Select Hour
            </label>
            <select value={hour} onChange={(e) => setHour(parseInt(e.target.value))} style={inputStyle}>
              {Array.from({ length: 24 }, (_, i) => (
                <option key={i} value={i} style={{ background: G.panelBg }}>
                  {i.toString().padStart(2, '0')}:00 — {i === 0 ? "Midnight" : i < 12 ? "Morning" : i === 12 ? "Noon" : i < 18 ? "Afternoon" : "Evening"}
                </option>
              ))}
            </select>
          </div>

          {/* Instructions */}
          {!selectedLocation && !loading && (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "32px", textAlign: "center" }}>
              <div>
                <div style={{ fontSize: "56px", marginBottom: "20px" }}>🗺️</div>
                <h3 style={{ margin: "0 0 10px 0", color: G.textPrimary, fontWeight: "700" }}>Click on the Map</h3>
                <p style={{ color: G.textMuted, fontSize: "14px", lineHeight: "1.7", margin: 0 }}>
                  Select any location in Sri Lanka to get wildlife conflict risk prediction and detailed analysis.
                </p>
              </div>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "32px" }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "40px", marginBottom: "16px", animation: "pulse 1.5s infinite" }}>🔍</div>
                <p style={{ color: G.textMuted, margin: 0, fontSize: "14px" }}>Analyzing location…</p>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{ margin: "16px", padding: "14px", background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "12px", color: "#f87171" }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Prediction Results */}
          {prediction && !loading && (
            <div style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "12px" }}>

              {/* Risk Level */}
              <div style={{
                background: getRiskBg(prediction.risk_level),
                border: `2px solid ${getRiskBorder(prediction.risk_level)}`,
                borderRadius: "16px", padding: "22px", textAlign: "center",
                backdropFilter: "blur(10px)",
              }}>
                <div style={{ fontSize: "40px", marginBottom: "10px" }}>
                  {prediction.risk_level === "High" ? "⚠️" : prediction.risk_level === "Medium" ? "⚡" : "✅"}
                </div>
                <h2 style={{ margin: "0 0 6px 0", color: getRiskColor(prediction.risk_level), fontSize: "24px", fontWeight: "800" }}>
                  {prediction.risk_level} Risk
                </h2>
                <p style={{ margin: 0, color: G.textMuted, fontSize: "13px" }}>
                  Conflict Risk Score: <strong style={{ color: G.textPrimary }}>{prediction.conflict_risk_score.toFixed(1)}</strong>
                </p>
              </div>

              {/* Probability */}
              <div style={{ background: G.card, backdropFilter: "blur(20px)", borderRadius: "14px", padding: "16px", border: `1px solid ${G.border}` }}>
                <h4 style={{ margin: "0 0 12px 0", fontSize: "13px", color: G.textSecondary, display: "flex", alignItems: "center", gap: "6px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                  🎯 Elephant Probability
                </h4>
                <div style={{ background: "rgba(255,255,255,0.08)", borderRadius: "8px", height: "10px", overflow: "hidden", marginBottom: "8px" }}>
                  <div style={{ width: `${prediction.elephant_probability * 100}%`, height: "100%", background: "linear-gradient(90deg, #10b981, #34d399)", transition: "width 0.5s ease" }} />
                </div>
                <p style={{ margin: 0, fontSize: "20px", fontWeight: "800", color: G.emeraldLight, textAlign: "center", letterSpacing: "-1px" }}>
                  {(prediction.elephant_probability * 100).toFixed(1)}%
                </p>
              </div>

              {/* Corridor Status */}
              <div style={{ background: G.card, backdropFilter: "blur(20px)", borderRadius: "14px", padding: "16px", border: `1px solid ${G.border}` }}>
                <h4 style={{ margin: "0 0 10px 0", fontSize: "13px", color: G.textSecondary, fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                  🛤️ Corridor Status
                </h4>
                <div style={{
                  display: "flex", alignItems: "center", padding: "12px",
                  background: prediction.near_corridor ? "rgba(245,158,11,0.1)" : "rgba(16,185,129,0.1)",
                  border: `1px solid ${prediction.near_corridor ? "rgba(245,158,11,0.3)" : "rgba(16,185,129,0.3)"}`,
                  borderRadius: "10px",
                }}>
                  <span style={{ fontSize: "22px", marginRight: "10px" }}>{prediction.near_corridor ? "⚠️" : "✅"}</span>
                  <span style={{ fontSize: "13px", color: G.textSecondary }}>
                    {prediction.near_corridor ? "Location is near an elephant corridor" : "Not near any known corridor"}
                  </span>
                </div>
              </div>

              {/* Nearest Node */}
              {prediction.nearest_node && (
                <div style={{ background: G.card, backdropFilter: "blur(20px)", borderRadius: "14px", padding: "16px", border: `1px solid ${G.border}` }}>
                  <h4 style={{ margin: "0 0 12px 0", fontSize: "13px", color: G.textSecondary, fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                    🐘 Nearest Elephant Hotspot
                  </h4>
                  <div style={{ fontSize: "13px", color: G.textSecondary }}>
                    {[
                      ["Node ID",            prediction.nearest_node.node_id],
                      ["Distance",           `${(prediction.distance_to_node / 1000).toFixed(2)} km`],
                      ["Elephant Count",     prediction.nearest_node.elephant_count],
                      ["Sightings",          prediction.nearest_node.sighting_count],
                      ["Active Hours",       prediction.nearest_node.active_hours.map(h => h + ':00').join(', ')],
                      ["NDVI",               prediction.nearest_node.avg_ndvi.toFixed(4)],
                      ["Avg Human Distance", `${(prediction.nearest_node.avg_human_distance / 1000).toFixed(2)} km`],
                      ["Protected Area",     prediction.nearest_node.protected ? "Yes" : "No"],
                    ].map(([k, v]) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${G.border}` }}>
                        <span>{k}</span>
                        <strong style={{ color: G.textPrimary }}>{v}</strong>
                      </div>
                    ))}
                    {prediction.nearest_node.elephants?.length > 0 && (
                      <div style={{ marginTop: "10px", padding: "10px", background: "rgba(255,255,255,0.04)", borderRadius: "8px" }}>
                        <span style={{ fontWeight: "600", color: G.textSecondary, fontSize: "12px" }}>Known Elephants:</span>
                        <div style={{ marginTop: "6px", display: "flex", flexWrap: "wrap", gap: "4px" }}>
                          {prediction.nearest_node.elephants.map((name, i) => (
                            <span key={i} style={{ background: "rgba(16,185,129,0.15)", border: "1px solid rgba(16,185,129,0.25)", color: "#34d399", padding: "3px 10px", borderRadius: "20px", fontSize: "12px" }}>
                              {name}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Nearest Corridor */}
              {prediction.nearest_corridor && (
                <div style={{ background: G.card, backdropFilter: "blur(20px)", borderRadius: "14px", padding: "16px", border: `1px solid ${G.border}` }}>
                  <h4 style={{ margin: "0 0 12px 0", fontSize: "13px", color: G.textSecondary, fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                    🛤️ Nearest Corridor
                  </h4>
                  <div style={{ fontSize: "13px", color: G.textSecondary }}>
                    {[
                      ["Corridor ID",        prediction.nearest_corridor.corridor_id],
                      ["Route",              `Node ${prediction.nearest_corridor.from_node} → ${prediction.nearest_corridor.to_node}`],
                      ["Distance to Corridor", `${(prediction.distance_to_corridor / 1000).toFixed(2)} km`],
                      ["Corridor Length",    `${(prediction.nearest_corridor.distance_meters / 1000).toFixed(2)} km`],
                      ["Usage Count",        `${prediction.nearest_corridor.usage_count}×`],
                      ["Crossing Count",     prediction.nearest_corridor.crossing_count],
                      ["Active Hours",       prediction.nearest_corridor.active_hours.map(h => h + ':00').join(', ')],
                      ["Avg Human Distance", `${(prediction.nearest_corridor.avg_human_distance / 1000).toFixed(2)} km`],
                      ["Bidirectional",      prediction.nearest_corridor.bidirectional ? "Yes" : "No"],
                    ].map(([k, v]) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${G.border}` }}>
                        <span>{k}</span>
                        <strong style={{ color: G.textPrimary }}>{v}</strong>
                      </div>
                    ))}
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0" }}>
                      <span>Safety Score</span>
                      <strong style={{ color: prediction.nearest_corridor.safety_score < 30 ? "#ef4444" : prediction.nearest_corridor.safety_score < 60 ? "#f59e0b" : "#10b981" }}>
                        {prediction.nearest_corridor.safety_score.toFixed(1)}
                      </strong>
                    </div>
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {prediction.recommendations?.length > 0 && (
                <div style={{ background: G.card, backdropFilter: "blur(20px)", borderRadius: "14px", padding: "16px", border: `1px solid ${G.border}` }}>
                  <h4 style={{ margin: "0 0 12px 0", fontSize: "13px", color: G.textSecondary, fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                    💡 Recommendations
                  </h4>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    {prediction.recommendations.map((rec, i) => (
                      <div key={i} style={{
                        display: "flex", alignItems: "flex-start", padding: "10px 12px",
                        background: "rgba(255,255,255,0.04)", borderRadius: "10px",
                        border: `1px solid ${G.border}`,
                      }}>
                        <span style={{ marginRight: "10px", fontSize: "16px", flexShrink: 0 }}>
                          {rec.includes("HIGH") || rec.includes("CRITICAL") ? "🚨" : rec.includes("MEDIUM") ? "⚠️" : "ℹ️"}
                        </span>
                        <span style={{ fontSize: "12px", color: G.textSecondary, lineHeight: "1.6" }}>{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Selected Coordinates */}
              {selectedLocation && (
                <div style={{ padding: "12px", background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)", borderRadius: "12px", fontSize: "12px", color: G.emeraldLight, textAlign: "center" }}>
                  <MapPin size={13} style={{ display: "inline", marginRight: "6px", verticalAlign: "middle" }} />
                  {selectedLocation.lat.toFixed(6)}, {selectedLocation.lng.toFixed(6)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Navigation Modal */}
      {showNavigationModal && navigationTarget && (
        <div style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          background: "rgba(0,0,0,0.7)", backdropFilter: "blur(6px)",
          display: "flex", alignItems: "center", justifyContent: "center", zIndex: 10000,
        }}>
          <div style={{
            background: "#111827", border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: "24px", boxShadow: "0 24px 64px rgba(0,0,0,0.6)",
            maxWidth: "480px", width: "90%", overflow: "hidden",
            animation: "slideIn 0.3s ease-out",
          }}>
            {/* Header */}
            <div style={{
              background: "linear-gradient(135deg, #064e3b, #065f46)",
              padding: "32px", textAlign: "center",
              borderBottom: "1px solid rgba(255,255,255,0.08)",
            }}>
              <div style={{ fontSize: "60px", marginBottom: "14px" }}>🗺️</div>
              <h2 style={{ margin: "0 0 8px 0", fontSize: "22px", fontWeight: "800", color: "#fff" }}>Location Too Far</h2>
              <p style={{ margin: 0, fontSize: "14px", color: "rgba(255,255,255,0.6)" }}>No elephant activity detected in this area</p>
            </div>

            {/* Content */}
            <div style={{ padding: "28px" }}>
              <div style={{ background: "rgba(245,158,11,0.12)", border: "2px solid rgba(245,158,11,0.35)", borderRadius: "14px", padding: "18px", marginBottom: "20px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <span style={{ fontSize: "22px" }}>📏</span>
                  <div>
                    <div style={{ fontSize: "12px", color: G.textMuted, marginBottom: "3px" }}>Distance to Nearest Activity Zone</div>
                    <div style={{ fontSize: "28px", fontWeight: "800", color: "#f59e0b", letterSpacing: "-1px" }}>{navigationTarget.distanceKm} km</div>
                  </div>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "22px" }}>
                <div style={{ background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)", borderRadius: "12px", padding: "16px", textAlign: "center" }}>
                  <div style={{ fontSize: "22px", marginBottom: "6px" }}>🐘</div>
                  <div style={{ fontSize: "11px", color: G.textMuted, marginBottom: "3px" }}>Nearest Zone</div>
                  <div style={{ fontSize: "15px", fontWeight: "700", color: G.emeraldLight }}>Node #{navigationTarget.nodeId}</div>
                </div>
                <div style={{ background: "rgba(96,165,250,0.1)", border: "1px solid rgba(96,165,250,0.2)", borderRadius: "12px", padding: "16px", textAlign: "center" }}>
                  <div style={{ fontSize: "22px", marginBottom: "6px" }}>✅</div>
                  <div style={{ fontSize: "11px", color: G.textMuted, marginBottom: "3px" }}>Current Risk</div>
                  <div style={{ fontSize: "15px", fontWeight: "700", color: "#60a5fa" }}>Low</div>
                </div>
              </div>

              <p style={{ fontSize: "14px", lineHeight: "1.7", color: G.textMuted, marginBottom: "22px", textAlign: "center" }}>
                This location is outside monitored elephant activity zones.
                Would you like to navigate to the nearest active zone?
              </p>

              <div style={{ display: "flex", gap: "12px" }}>
                <button onClick={() => setShowNavigationModal(false)} style={{
                  flex: 1, padding: "13px 18px", background: "rgba(255,255,255,0.06)",
                  color: G.textSecondary, border: `1px solid ${G.border}`,
                  borderRadius: "12px", fontSize: "14px", fontWeight: "600", cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: "8px",
                  transition: "all 0.2s",
                }}
                  onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.1)"}
                  onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.06)"}
                >
                  <MapPin size={15} /> Stay Here
                </button>
                <button onClick={handleNavigateToZone} style={{
                  flex: 1, padding: "13px 18px",
                  background: "linear-gradient(135deg, #059669, #10b981)",
                  color: "white", border: "none", borderRadius: "12px",
                  fontSize: "14px", fontWeight: "700", cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: "8px",
                  boxShadow: "0 4px 16px rgba(16,185,129,0.35)", transition: "all 0.2s",
                }}
                  onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "0 6px 20px rgba(16,185,129,0.45)"; }}
                  onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)"; e.currentTarget.style.boxShadow = "0 4px 16px rgba(16,185,129,0.35)"; }}
                >
                  <Navigation size={15} /> Navigate to Zone
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
