import { useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, Circle, Polyline, useMapEvents } from "react-leaflet";
import { Link } from "react-router-dom";
import axios from "axios";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Custom pin icon for location selection
const pinIcon = L.divIcon({
  html: '<div style="font-size: 28px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📍</div>',
  className: 'custom-pin-icon',
  iconSize: [28, 28],
  iconAnchor: [14, 28],
  popupAnchor: [0, -28]
});

// Elephant icon for nearest node
const elephantIcon = L.divIcon({
  html: '<div style="font-size: 24px;">🐘</div>',
  className: 'custom-elephant-icon',
  iconSize: [24, 24],
  iconAnchor: [12, 12],
  popupAnchor: [0, -12]
});

// Location picker component
function LocationPicker({ onLocationSelect }) {
  useMapEvents({
    click: (e) => {
      onLocationSelect(e.latlng);
    }
  });
  return null;
}

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
      // Create datetime with selected date and hour
      const dateTime = new Date(selectedDate);
      dateTime.setHours(hour, 0, 0, 0);
      const datetime = dateTime.toISOString();

      const response = await axios.post(`http://localhost:8000/predict_risk`, {
        latitude: latlng.lat,
        longitude: latlng.lng,
        datetime: datetime
      });
      
      const result = response.data;
      setPrediction(result);
      
      // Check if location is too far from elephant activity
      if (result.distance_to_node > 25000) {
        // Show custom navigation modal for distant locations
        const distanceKm = (result.distance_to_node / 1000).toFixed(1);
        setNavigationTarget({
          distanceKm,
          targetLat: result.nearest_node.center_lat,
          targetLon: result.nearest_node.center_lon,
          nodeId: result.nearest_node.node_id
        });
        setShowNavigationModal(true);
      }
    } catch (err) {
      console.error("Error fetching prediction:", err);
      setError(err.response?.data?.detail || err.message || "Failed to fetch prediction");
    } finally {
      setLoading(false);
    }
  };

  const handleNavigateToZone = () => {
    if (navigationTarget) {
      setSelectedLocation({
        lat: navigationTarget.targetLat,
        lng: navigationTarget.targetLon
      });
      setShowNavigationModal(false);
      // Re-fetch prediction for new location
      handleLocationSelect({
        lat: navigationTarget.targetLat,
        lng: navigationTarget.targetLon
      });
    }
  };

  const handleStayHere = () => {
    setShowNavigationModal(false);
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case "High": return "#d32f2f";
      case "Medium": return "#f57c00";
      case "Low": return "#388e3c";
      default: return "#757575";
    }
  };

  const getRiskBgColor = (riskLevel) => {
    switch (riskLevel) {
      case "High": return "#ffebee";
      case "Medium": return "#fff3e0";
      case "Low": return "#e8f5e9";
      default: return "#f5f5f5";
    }
  };

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      {/** */} <div style={{
        backgroundColor: "#1a237e",
        color: "white",
        padding: "15px 20px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        boxShadow: "0 2px 10px rgba(0,0,0,0.2)"
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "20px", fontWeight: "600" }}>
            🐘 Wildlife Conflict Prediction
          </h1>
          <p style={{ margin: "5px 0 0 0", fontSize: "12px", opacity: 0.8 }}>
            Click anywhere on the map to predict elephant activity risk
          </p>
        </div>
        <Link 
          to="/" 
          style={{
            backgroundColor: "rgba(255,255,255,0.2)",
            color: "white",
            padding: "10px 20px",
            borderRadius: "6px",
            textDecoration: "none",
            fontSize: "14px",
            fontWeight: "500",
            transition: "background-color 0.2s"
          }}
        >
          ← Back to Map
        </Link>
      </div>

      <div style={{ flex: 1, display: "flex", position: "relative" }}>
        {/* Map Container */}
        <div style={{ flex: 1 }}>
          <MapContainer
            center={[7.8731, 80.7718]}
            zoom={8}
            style={{ height: "100%", width: "100%" }}
            maxBounds={[[5.8, 79.4], [9.9, 82.0]]}
            minZoom={7}
            maxZoom={15}
            maxBoundsViscosity={1.0}
          >
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            <LocationPicker onLocationSelect={handleLocationSelect} />

            {/* Selected location marker */}
            {selectedLocation && (
              <Marker position={[selectedLocation.lat, selectedLocation.lng]} icon={pinIcon}>
                <Popup>
                  <div style={{ textAlign: "center" }}>
                    <strong>Selected Location</strong><br />
                    Lat: {selectedLocation.lat.toFixed(6)}<br />
                    Lon: {selectedLocation.lng.toFixed(6)}
                  </div>
                </Popup>
              </Marker>
            )}

            {/* Nearest node visualization */}
            {prediction && prediction.nearest_node && (
              <>
                <Circle
                  center={[prediction.nearest_node.center_lat, prediction.nearest_node.center_lon]}
                  radius={prediction.nearest_node.radius_meters}
                  pathOptions={{
                    color: "#ff9800",
                    weight: 2,
                    fillColor: "#ff9800",
                    fillOpacity: 0.2
                  }}
                />
                <Marker 
                  position={[prediction.nearest_node.center_lat, prediction.nearest_node.center_lon]} 
                  icon={elephantIcon}
                >
                  <Popup>
                    <div>
                      <strong>🐘 Nearest Hotspot</strong><br />
                      Node ID: {prediction.nearest_node.node_id}<br />
                      Elephants: {prediction.nearest_node.elephant_count}<br />
                      Distance: {(prediction.distance_to_node / 1000).toFixed(2)} km
                    </div>
                  </Popup>
                </Marker>
              </>
            )}

            {/* Nearest corridor visualization */}
            {prediction && prediction.nearest_corridor && (
              <Polyline
                positions={prediction.nearest_corridor.path.map(p => [p.lat, p.lon])}
                pathOptions={{
                  color: "#9c27b0",
                  weight: 4,
                  opacity: 0.7,
                  dashArray: "10, 8"
                }}
              >
                <Popup>
                  <div>
                    <strong>🛤️ Nearest Corridor</strong><br />
                    ID: {prediction.nearest_corridor.corridor_id}<br />
                    Distance: {(prediction.distance_to_corridor / 1000).toFixed(2)} km
                  </div>
                </Popup>
              </Polyline>
            )}

            {/* Connection line from selected point to nearest node */}
            {selectedLocation && prediction && prediction.nearest_node && (
              <Polyline
                positions={[
                  [selectedLocation.lat, selectedLocation.lng],
                  [prediction.nearest_node.center_lat, prediction.nearest_node.center_lon]
                ]}
                pathOptions={{
                  color: "#607d8b",
                  weight: 2,
                  opacity: 0.5,
                  dashArray: "5, 10"
                }}
              />
            )}
          </MapContainer>
        </div>

        {/* Side Panel */}
        <div style={{
          width: "400px",
          backgroundColor: "#f8f9fa",
          overflowY: "auto",
          borderLeft: "1px solid #e0e0e0",
          display: "flex",
          flexDirection: "column"
        }}>
          {/* Date & Time Selector */}
          <div style={{
            padding: "15px",
            backgroundColor: "white",
            borderBottom: "1px solid #e0e0e0"
          }}>
            {/* Date Picker */}
            <label style={{ fontWeight: "600", fontSize: "14px", display: "block", marginBottom: "8px" }}>
              📅 Select Date
            </label>
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              style={{
                width: "100%",
                padding: "10px",
                fontSize: "14px",
                borderRadius: "6px",
                border: "1px solid #ccc",
                cursor: "pointer",
                marginBottom: "15px",
                boxSizing: "border-box"
              }}
            />

            {/* Hour Selector */}
            <label style={{ fontWeight: "600", fontSize: "14px", display: "block", marginBottom: "8px" }}>
              🕐 Select Hour
            </label>
            <select
              value={hour}
              onChange={(e) => setHour(parseInt(e.target.value))}
              style={{
                width: "100%",
                padding: "10px",
                fontSize: "14px",
                borderRadius: "6px",
                border: "1px solid #ccc",
                cursor: "pointer"
              }}
            >
              {Array.from({ length: 24 }, (_, i) => (
                <option key={i} value={i}>
                  {i.toString().padStart(2, '0')}:00 - {i === 0 ? "Midnight" : i < 12 ? "Morning" : i === 12 ? "Noon" : i < 18 ? "Afternoon" : "Evening"}
                </option>
              ))}
            </select>
          </div>

          {/* Instructions */}
          {!selectedLocation && !loading && (
            <div style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "30px",
              textAlign: "center"
            }}>
              <div>
                <div style={{ fontSize: "60px", marginBottom: "20px" }}>🗺️</div>
                <h3 style={{ margin: "0 0 10px 0", color: "#333" }}>Click on the Map</h3>
                <p style={{ color: "#666", fontSize: "14px", lineHeight: "1.6" }}>
                  Select any location in Sri Lanka to get wildlife conflict risk prediction and detailed analysis.
                </p>
              </div>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "30px"
            }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "40px", marginBottom: "15px", animation: "pulse 1.5s infinite" }}>🔍</div>
                <p style={{ color: "#666" }}>Analyzing location...</p>
              </div>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div style={{
              margin: "15px",
              padding: "15px",
              backgroundColor: "#ffebee",
              borderRadius: "8px",
              color: "#c62828"
            }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Prediction Results */}
          {prediction && !loading && (
            <div style={{ padding: "15px" }}>
              {/* Risk Level Card */}
              <div style={{
                backgroundColor: getRiskBgColor(prediction.risk_level),
                borderRadius: "12px",
                padding: "20px",
                marginBottom: "15px",
                border: `2px solid ${getRiskColor(prediction.risk_level)}`,
                textAlign: "center"
              }}>
                <div style={{ fontSize: "40px", marginBottom: "10px" }}>
                  {prediction.risk_level === "High" ? "⚠️" : prediction.risk_level === "Medium" ? "⚡" : "✅"}
                </div>
                <h2 style={{ 
                  margin: "0 0 5px 0", 
                  color: getRiskColor(prediction.risk_level),
                  fontSize: "24px"
                }}>
                  {prediction.risk_level} Risk
                </h2>
                <p style={{ margin: 0, color: "#666", fontSize: "13px" }}>
                  Conflict Risk Score: <strong>{prediction.conflict_risk_score.toFixed(1)}</strong>
                </p>
              </div>

              {/* Probability Card */}
              <div style={{
                backgroundColor: "white",
                borderRadius: "10px",
                padding: "15px",
                marginBottom: "15px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
              }}>
                <h4 style={{ margin: "0 0 10px 0", fontSize: "14px", color: "#333" }}>
                  🎯 Elephant Probability
                </h4>
                <div style={{
                  backgroundColor: "#e3f2fd",
                  borderRadius: "8px",
                  height: "20px",
                  overflow: "hidden",
                  marginBottom: "8px"
                }}>
                  <div style={{
                    width: `${prediction.elephant_probability * 100}%`,
                    height: "100%",
                    backgroundColor: "#1976d2",
                    transition: "width 0.5s ease"
                  }} />
                </div>
                <p style={{ margin: 0, fontSize: "18px", fontWeight: "600", color: "#1976d2", textAlign: "center" }}>
                  {(prediction.elephant_probability * 100).toFixed(1)}%
                </p>
              </div>

              {/* Corridor Status */}
              <div style={{
                backgroundColor: "white",
                borderRadius: "10px",
                padding: "15px",
                marginBottom: "15px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
              }}>
                <h4 style={{ margin: "0 0 10px 0", fontSize: "14px", color: "#333" }}>
                  🛤️ Corridor Status
                </h4>
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  padding: "10px",
                  backgroundColor: prediction.near_corridor ? "#fff3e0" : "#e8f5e9",
                  borderRadius: "6px"
                }}>
                  <span style={{ fontSize: "24px", marginRight: "10px" }}>
                    {prediction.near_corridor ? "⚠️" : "✅"}
                  </span>
                  <span style={{ fontSize: "13px", color: "#555" }}>
                    {prediction.near_corridor 
                      ? "Location is near an elephant corridor" 
                      : "Not near any known corridor"}
                  </span>
                </div>
              </div>

              {/* Nearest Node Details */}
              {prediction.nearest_node && (
                <div style={{
                  backgroundColor: "white",
                  borderRadius: "10px",
                  padding: "15px",
                  marginBottom: "15px",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                }}>
                  <h4 style={{ margin: "0 0 12px 0", fontSize: "14px", color: "#333" }}>
                    🐘 Nearest Elephant Hotspot
                  </h4>
                  <div style={{ fontSize: "13px", lineHeight: "1.8", color: "#555" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Node ID:</span>
                      <strong>{prediction.nearest_node.node_id}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Distance:</span>
                      <strong>{(prediction.distance_to_node / 1000).toFixed(2)} km</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Elephant Count:</span>
                      <strong>{prediction.nearest_node.elephant_count}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Sightings:</span>
                      <strong>{prediction.nearest_node.sighting_count}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Active Hours:</span>
                      <strong>{prediction.nearest_node.active_hours.map(h => h + ':00').join(', ')}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>NDVI:</span>
                      <strong>{prediction.nearest_node.avg_ndvi.toFixed(4)}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Avg Human Distance:</span>
                      <strong>{(prediction.nearest_node.avg_human_distance / 1000).toFixed(2)} km</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0" }}>
                      <span>Protected Area:</span>
                      <strong>{prediction.nearest_node.protected ? "Yes" : "No"}</strong>
                    </div>
                    {prediction.nearest_node.elephants && prediction.nearest_node.elephants.length > 0 && (
                      <div style={{ marginTop: "10px", padding: "10px", backgroundColor: "#f5f5f5", borderRadius: "6px" }}>
                        <span style={{ fontWeight: "600" }}>Known Elephants:</span>
                        <div style={{ marginTop: "5px" }}>
                          {prediction.nearest_node.elephants.map((name, i) => (
                            <span key={i} style={{
                              display: "inline-block",
                              backgroundColor: "#e3f2fd",
                              padding: "3px 8px",
                              borderRadius: "12px",
                              fontSize: "12px",
                              margin: "2px"
                            }}>
                              {name}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Nearest Corridor Details */}
              {prediction.nearest_corridor && (
                <div style={{
                  backgroundColor: "white",
                  borderRadius: "10px",
                  padding: "15px",
                  marginBottom: "15px",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                }}>
                  <h4 style={{ margin: "0 0 12px 0", fontSize: "14px", color: "#333" }}>
                    🛤️ Nearest Corridor
                  </h4>
                  <div style={{ fontSize: "13px", lineHeight: "1.8", color: "#555" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Corridor ID:</span>
                      <strong>{prediction.nearest_corridor.corridor_id}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Route:</span>
                      <strong>Node {prediction.nearest_corridor.from_node} → {prediction.nearest_corridor.to_node}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Distance to Corridor:</span>
                      <strong>{(prediction.distance_to_corridor / 1000).toFixed(2)} km</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Corridor Length:</span>
                      <strong>{(prediction.nearest_corridor.distance_meters / 1000).toFixed(2)} km</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Usage Count:</span>
                      <strong>{prediction.nearest_corridor.usage_count}×</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Crossing Count:</span>
                      <strong>{prediction.nearest_corridor.crossing_count}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Active Hours:</span>
                      <strong>{prediction.nearest_corridor.active_hours.map(h => h + ':00').join(', ')}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Safety Score:</span>
                      <strong style={{ color: prediction.nearest_corridor.safety_score < 30 ? "#d32f2f" : prediction.nearest_corridor.safety_score < 60 ? "#f57c00" : "#388e3c" }}>
                        {prediction.nearest_corridor.safety_score.toFixed(1)}
                      </strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #f0f0f0" }}>
                      <span>Avg Human Distance:</span>
                      <strong>{(prediction.nearest_corridor.avg_human_distance / 1000).toFixed(2)} km</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0" }}>
                      <span>Bidirectional:</span>
                      <strong>{prediction.nearest_corridor.bidirectional ? "Yes" : "No"}</strong>
                    </div>
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {prediction.recommendations && prediction.recommendations.length > 0 && (
                <div style={{
                  backgroundColor: "white",
                  borderRadius: "10px",
                  padding: "15px",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                }}>
                  <h4 style={{ margin: "0 0 12px 0", fontSize: "14px", color: "#333" }}>
                    💡 Recommendations
                  </h4>
                  <div>
                    {prediction.recommendations.map((rec, i) => (
                      <div key={i} style={{
                        display: "flex",
                        alignItems: "flex-start",
                        padding: "10px",
                        backgroundColor: i % 2 === 0 ? "#f8f9fa" : "white",
                        borderRadius: "6px",
                        marginBottom: "5px"
                      }}>
                        <span style={{ marginRight: "10px", fontSize: "16px" }}>
                          {rec.includes("HIGH") || rec.includes("CRITICAL") ? "🚨" : 
                           rec.includes("MEDIUM") ? "⚠️" : "ℹ️"}
                        </span>
                        <span style={{ fontSize: "13px", color: "#555", lineHeight: "1.5" }}>{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Selected Coordinates */}
              {selectedLocation && (
                <div style={{
                  marginTop: "15px",
                  padding: "12px",
                  backgroundColor: "#e8eaf6",
                  borderRadius: "8px",
                  fontSize: "12px",
                  color: "#3f51b5",
                  textAlign: "center"
                }}>
                  📍 Selected: {selectedLocation.lat.toFixed(6)}, {selectedLocation.lng.toFixed(6)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* CSS Animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(-20px) scale(0.95);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
      `}</style>

      {/* Custom Navigation Modal */}
      {showNavigationModal && navigationTarget && (
        <div style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: "rgba(0, 0, 0, 0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 10000,
          backdropFilter: "blur(4px)"
        }}>
          <div style={{
            backgroundColor: "white",
            borderRadius: "20px",
            boxShadow: "0 20px 60px rgba(0, 0, 0, 0.3)",
            maxWidth: "500px",
            width: "90%",
            overflow: "hidden",
            animation: "slideIn 0.3s ease-out"
          }}>
            {/* Header */}
            <div style={{
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              padding: "30px",
              textAlign: "center",
              color: "white"
            }}>
              <div style={{ fontSize: "64px", marginBottom: "15px" }}>🗺️</div>
              <h2 style={{ margin: "0 0 10px 0", fontSize: "24px", fontWeight: "600" }}>
                Location Too Far
              </h2>
              <p style={{ margin: 0, fontSize: "14px", opacity: 0.9 }}>
                No elephant activity detected in this area
              </p>
            </div>

            {/* Content */}
            <div style={{ padding: "30px" }}>
              {/* Distance Info */}
              <div style={{
                backgroundColor: "#fff3e0",
                borderRadius: "12px",
                padding: "20px",
                marginBottom: "20px",
                border: "2px solid #ff9800"
              }}>
                <div style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
                  <span style={{ fontSize: "24px", marginRight: "10px" }}>📏</span>
                  <div>
                    <div style={{ fontSize: "12px", color: "#666", marginBottom: "3px" }}>
                      Distance to Nearest Activity Zone
                    </div>
                    <div style={{ fontSize: "28px", fontWeight: "700", color: "#f57c00" }}>
                      {navigationTarget.distanceKm} km
                    </div>
                  </div>
                </div>
              </div>

              {/* Info Cards */}
              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "12px",
                marginBottom: "25px"
              }}>
                <div style={{
                  backgroundColor: "#e8f5e9",
                  borderRadius: "10px",
                  padding: "15px",
                  textAlign: "center"
                }}>
                  <div style={{ fontSize: "24px", marginBottom: "5px" }}>🐘</div>
                  <div style={{ fontSize: "11px", color: "#666", marginBottom: "3px" }}>Nearest Zone</div>
                  <div style={{ fontSize: "16px", fontWeight: "600", color: "#2e7d32" }}>
                    Node #{navigationTarget.nodeId}
                  </div>
                </div>
                <div style={{
                  backgroundColor: "#e3f2fd",
                  borderRadius: "10px",
                  padding: "15px",
                  textAlign: "center"
                }}>
                  <div style={{ fontSize: "24px", marginBottom: "5px" }}>✅</div>
                  <div style={{ fontSize: "11px", color: "#666", marginBottom: "3px" }}>Current Risk</div>
                  <div style={{ fontSize: "16px", fontWeight: "600", color: "#1976d2" }}>
                    Low
                  </div>
                </div>
              </div>

              {/* Description */}
              <p style={{
                fontSize: "14px",
                lineHeight: "1.6",
                color: "#555",
                marginBottom: "25px",
                textAlign: "center"
              }}>
                This location is outside the monitored elephant activity zones. 
                Would you like to navigate to the nearest active zone for detailed predictions?
              </p>

              {/* Action Buttons */}
              <div style={{ display: "flex", gap: "12px" }}>
                <button
                  onClick={handleStayHere}
                  style={{
                    flex: 1,
                    padding: "14px 20px",
                    backgroundColor: "#f5f5f5",
                    color: "#555",
                    border: "2px solid #e0e0e0",
                    borderRadius: "10px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    transition: "all 0.2s",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "8px"
                  }}
                  onMouseOver={(e) => {
                    e.target.style.backgroundColor = "#eeeeee";
                    e.target.style.transform = "translateY(-2px)";
                  }}
                  onMouseOut={(e) => {
                    e.target.style.backgroundColor = "#f5f5f5";
                    e.target.style.transform = "translateY(0)";
                  }}
                >
                  <span>📍</span>
                  Stay Here
                </button>
                <button
                  onClick={handleNavigateToZone}
                  style={{
                    flex: 1,
                    padding: "14px 20px",
                    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "10px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    transition: "all 0.2s",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "8px",
                    boxShadow: "0 4px 12px rgba(102, 126, 234, 0.4)"
                  }}
                  onMouseOver={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow = "0 6px 16px rgba(102, 126, 234, 0.5)";
                  }}
                  onMouseOut={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow = "0 4px 12px rgba(102, 126, 234, 0.4)";
                  }}
                >
                  <span>🧭</span>
                  Navigate to Zone
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
