import { useState, useEffect } from "react";
import { MapContainer, TileLayer, Circle, Marker, Popup, Polyline, useMapEvents } from "react-leaflet";
import axios from "axios";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Function to determine district based on coordinates (approximate boundaries)
const getDistrict = (lat, lon) => {
  // Southern Province
  if (lat >= 6.0 && lat <= 6.5 && lon >= 80.0 && lon <= 81.5) {
    if (lon < 80.5) return "Galle";
    if (lon < 81.0) return "Matara";
    return "Hambantota";
  }
  // Uva Province
  if (lat >= 6.5 && lat <= 7.5 && lon >= 81.0 && lon <= 81.8) {
    if (lat < 6.8) return "Monaragala";
    return "Badulla";
  }
  // Sabaragamuwa Province
  if (lat >= 6.5 && lat <= 7.5 && lon >= 80.2 && lon <= 81.0) {
    if (lat < 7.0) return "Ratnapura";
    return "Kegalle";
  }
  // Central Province
  if (lat >= 7.0 && lat <= 7.5 && lon >= 80.5 && lon <= 81.3) {
    if (lon < 80.8) return "Kandy";
    if (lat < 7.3) return "Matale";
    return "Nuwara Eliya";
  }
  // North Central Province
  if (lat >= 7.5 && lat <= 8.5 && lon >= 80.0 && lon <= 81.3) {
    if (lon < 80.5) return "Anuradhapura";
    return "Polonnaruwa";
  }
  // Eastern Province
  if (lat >= 7.0 && lon >= 81.0) {
    if (lat < 7.5) return "Ampara";
    if (lat < 8.5) return "Batticaloa";
    return "Trincomalee";
  }
  // Western Province
  if (lat >= 6.5 && lat <= 7.5 && lon >= 79.5 && lon <= 80.5) {
    if (lat > 7.0 && lon > 79.8 && lon < 80.2) return "Colombo";
    if (lat < 7.0) return "Kalutara";
    return "Gampaha";
  }
  // North Western Province
  if (lat >= 7.0 && lat <= 8.5 && lon >= 79.5 && lon <= 80.5) {
    if (lat < 7.8) return "Kurunegala";
    return "Puttalam";
  }
  
  return "Other";
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

  // Component to track zoom level
  function ZoomTracker() {
    useMapEvents({
      zoomend: (e) => {
        setZoom(e.target.getZoom());
      }
    });
    return null;
  }

  // Create custom elephant icon
  const elephantIcon = L.divIcon({
    html: '<div style="font-size: 16px; text-align: center;">🐘</div>',
    className: 'custom-elephant-icon',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
    popupAnchor: [0, -10]
  });

  useEffect(() => {
    console.log("Attempting to fetch nodes from API...");
    setLoading(true);
    
    // Fetch nodes
    axios.get("http://localhost:8000/nodes")
      .then((response) => {
        console.log("Fetched nodes:", response.data);
        const nodesWithDistricts = response.data.map(node => ({
          ...node,
          district: getDistrict(node.center_lat, node.center_lon)
        }));
        setNodes(nodesWithDistricts);
        
        // Extract unique districts
        const uniqueDistricts = [...new Set(nodesWithDistricts.map(n => n.district))].sort();
        setDistricts(uniqueDistricts);
        
        setError(null);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching nodes:", error);
        console.error("Error details:", error.response || error.message);
        setError(error.message);
        setLoading(false);
      });
    
    // Fetch corridors
    axios.get("http://localhost:8000/corridors")
      .then((response) => {
        console.log("Fetched corridors:", response.data);
        setCorridors(response.data);
      })
      .catch((error) => {
        console.error("Error fetching corridors:", error);
      });
  }, []);

  console.log("Current nodes state:", nodes);

  // Get danger level category
  const getDangerLevel = (score) => {
    if (score < 30) return "High";
    if (score < 60) return "Medium";
    return "Low";
  };

  // Filter nodes by all criteria
  const filteredNodes = nodes.filter(n => {
    // District filter
    if (selectedDistrict !== "All" && n.district !== selectedDistrict) return false;
    
    // Danger level filter
    if (dangerLevel !== "All") {
      const nodeDanger = getDangerLevel(n.safety_score);
      if (nodeDanger !== dangerLevel) return false;
    }
    
    // Elephant count filter
    if (elephantCountFilter !== "All") {
      if (elephantCountFilter === "1" && n.elephant_count !== 1) return false;
      if (elephantCountFilter === "2" && n.elephant_count !== 2) return false;
      if (elephantCountFilter === "3+" && n.elephant_count < 3) return false;
    }
    
    // Time filter (active hours)
    if (timeFilter !== "All") {
      const hour = parseInt(timeFilter);
      if (!n.active_hours.includes(hour)) return false;
    }
    
    return true;
  });

  const getColor = (safety_score) => {
    if (safety_score < 30) return "#d32f2f";  // Deep Red - High danger
    if (safety_score < 60) return "#f57c00";  // Deep Orange - Medium danger
    if (safety_score < 80) return "#fbc02d";  // Yellow - Moderate
    return "#388e3c";  // Green - Low danger
  };

  const getOpacity = (elephant_count) => {
    return Math.min(0.2 + (elephant_count * 0.1), 0.7);
  };

  const getCorridorWidth = (usage_count) => {
    return Math.min(3 + usage_count, 8);
  };

  const getCorridorOpacity = (safety_score) => {
    if (safety_score < 30) return 0.85;
    if (safety_score < 60) return 0.75;
    return 0.65;
  };

  return (
    <div>
      {/* Error message */}
      {error && (
        <div style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          zIndex: 2000,
          backgroundColor: "#ff4444",
          color: "white",
          padding: "20px",
          borderRadius: "8px",
          boxShadow: "0 2px 10px rgba(0,0,0,0.3)",
          maxWidth: "400px"
        }}>
          <h3 style={{ margin: "0 0 10px 0" }}>API Connection Error</h3>
          <p style={{ margin: 0 }}>{error}</p>
          <p style={{ margin: "10px 0 0 0", fontSize: "12px" }}>
            Make sure the API server is running on http://localhost:8000
          </p>
        </div>
      )}

      {/* Loading message */}
      {loading && !error && (
        <div style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          zIndex: 2000,
          backgroundColor: "white",
          padding: "20px",
          borderRadius: "8px",
          boxShadow: "0 2px 10px rgba(0,0,0,0.3)"
        }}>
          <p style={{ margin: 0 }}>Loading elephant data...</p>
        </div>
      )}

      <MapContainer
        center={[7.8731, 80.7718]}   // Sri Lanka center
        zoom={8}
        style={{ height: "100vh", width: "100%" }}
        maxBounds={[[5.5, 79.0], [10.0, 82.5]]}
        minZoom={7}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <ZoomTracker />

        {/* Filter Panel - Right Side */}
        <div style={{
          position: "absolute",
          top: "10px",
          right: "10px",
          zIndex: 1000,
          backgroundColor: "white",
          padding: "20px",
          borderRadius: "8px",
          boxShadow: "0 2px 10px rgba(0,0,0,0.3)",
          maxWidth: "280px",
          maxHeight: "calc(100vh - 20px)",
          overflowY: "auto"
        }}>
          <h3 style={{ margin: "0 0 15px 0", fontSize: "16px", fontWeight: "bold" }}>
            Filters
          </h3>

          {/* Results Count */}
          <div style={{ 
            marginBottom: "15px", 
            padding: "10px", 
            backgroundColor: "#f0f0f0", 
            borderRadius: "4px",
            textAlign: "center"
          }}>
            <strong>{filteredNodes.length}</strong> location{filteredNodes.length !== 1 ? 's' : ''}
          </div>

          {/* Show Corridors Toggle */}
          <div style={{ marginBottom: "15px" }}>
            <label style={{ fontSize: "13px", display: "flex", alignItems: "center", cursor: "pointer" }}>
              <input 
                type="checkbox" 
                checked={showCorridors} 
                onChange={(e) => setShowCorridors(e.target.checked)}
                style={{ marginRight: "8px", cursor: "pointer" }}
              />
              Show Elephant Corridors
            </label>
          </div>

          {/* Danger Level Filter */}
          <div style={{ marginBottom: "15px" }}>
            <label style={{ fontSize: "13px", fontWeight: "bold", display: "block", marginBottom: "5px" }}>
              Filter by Danger Level
            </label>
            <select 
              value={dangerLevel} 
              onChange={(e) => setDangerLevel(e.target.value)}
              style={{
                width: "100%",
                padding: "8px",
                fontSize: "13px",
                borderRadius: "4px",
                border: "1px solid #ccc",
                cursor: "pointer"
              }}
            >
              <option value="All">All Levels</option>
              <option value="High">🔴 High Danger (&lt;30)</option>
              <option value="Medium">🟠 Medium Danger (30-60)</option>
              <option value="Low">🟢 Low Danger (&gt;60)</option>
            </select>
          </div>

          {/* Elephant Count Filter */}
          <div style={{ marginBottom: "15px" }}>
            <label style={{ fontSize: "13px", fontWeight: "bold", display: "block", marginBottom: "5px" }}>
              Filter by Elephant Count
            </label>
            <select 
              value={elephantCountFilter} 
              onChange={(e) => setElephantCountFilter(e.target.value)}
              style={{
                width: "100%",
                padding: "8px",
                fontSize: "13px",
                borderRadius: "4px",
                border: "1px solid #ccc",
                cursor: "pointer"
              }}
            >
              <option value="All">All Counts</option>
              <option value="1">1 Elephant</option>
              <option value="2">2 Elephants</option>
              <option value="3+">3+ Elephants</option>
            </select>
          </div>

          {/* District Filter */}
          <div style={{ marginBottom: "15px" }}>
            <label style={{ fontSize: "13px", fontWeight: "bold", display: "block", marginBottom: "5px" }}>
              Filter by Region / District
            </label>
            <select 
              value={selectedDistrict} 
              onChange={(e) => setSelectedDistrict(e.target.value)}
              style={{
                width: "100%",
                padding: "8px",
                fontSize: "13px",
                borderRadius: "4px",
                border: "1px solid #ccc",
                cursor: "pointer"
              }}
            >
              <option value="All">All Districts</option>
              {districts.map(district => (
                <option key={district} value={district}>{district}</option>
              ))}
            </select>
          </div>

          {/* Time Period Filter */}
          <div style={{ marginBottom: "15px" }}>
            <label style={{ fontSize: "13px", fontWeight: "bold", display: "block", marginBottom: "5px" }}>
              Filter by Time Period (Active Hours)
            </label>
            <select 
              value={timeFilter} 
              onChange={(e) => setTimeFilter(e.target.value)}
              style={{
                width: "100%",
                padding: "8px",
                fontSize: "13px",
                borderRadius: "4px",
                border: "1px solid #ccc",
                cursor: "pointer"
              }}
            >
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
              setDangerLevel("All");
              setElephantCountFilter("All");
              setSelectedDistrict("All");
              setTimeFilter("All");
            }}
            style={{
              width: "100%",
              padding: "10px",
              fontSize: "13px",
              fontWeight: "bold",
              backgroundColor: "#ff6b6b",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Reset All Filters
          </button>
        </div>

      {/* Legend */}
      <div style={{
        position: "absolute",
        bottom: "20px",
        left: "10px",
        zIndex: 1000,
        backgroundColor: "rgba(255, 255, 255, 0.95)",
        padding: "15px",
        borderRadius: "10px",
        boxShadow: "0 4px 12px rgba(0,0,0,0.25)",
        backdropFilter: "blur(5px)",
        border: "1px solid rgba(0,0,0,0.1)"
      }}>
        <h4 style={{ margin: "0 0 12px 0", fontSize: "14px", fontWeight: "600", color: "#333" }}>Safety Levels</h4>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "6px" }}>
          <div style={{ width: "20px", height: "20px", backgroundColor: "#d32f2f", marginRight: "10px", borderRadius: "50%", boxShadow: "0 2px 4px rgba(0,0,0,0.2)" }}></div>
          <span style={{ fontSize: "12px", color: "#555" }}>High Risk (&lt;30)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "6px" }}>
          <div style={{ width: "20px", height: "20px", backgroundColor: "#f57c00", marginRight: "10px", borderRadius: "50%", boxShadow: "0 2px 4px rgba(0,0,0,0.2)" }}></div>
          <span style={{ fontSize: "12px", color: "#555" }}>Medium (30-60)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "6px" }}>
          <div style={{ width: "20px", height: "20px", backgroundColor: "#fbc02d", marginRight: "10px", borderRadius: "50%", boxShadow: "0 2px 4px rgba(0,0,0,0.2)" }}></div>
          <span style={{ fontSize: "12px", color: "#555" }}>Moderate (60-80)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "12px" }}>
          <div style={{ width: "20px", height: "20px", backgroundColor: "#388e3c", marginRight: "10px", borderRadius: "50%", boxShadow: "0 2px 4px rgba(0,0,0,0.2)" }}></div>
          <span style={{ fontSize: "12px", color: "#555" }}>Low Risk (&gt;80)</span>
        </div>
        <div style={{ borderTop: "1px solid #e0e0e0", paddingTop: "10px", marginTop: "10px" }}>
          <div style={{ fontSize: "11px", color: "#666", marginBottom: "4px" }}>🐘 Hotspot Zones</div>
          <div style={{ fontSize: "11px", color: "#666", display: "flex", alignItems: "center" }}>
            <div style={{ width: "30px", height: "3px", background: "linear-gradient(90deg, #d32f2f 0%, #d32f2f 50%, transparent 50%, transparent 100%)", backgroundSize: "10px 3px", marginRight: "8px" }}></div>
            <span>Corridors</span>
          </div>
        </div>
      </div>
      {/* Draw Corridors as Polylines */}
      {showCorridors && corridors.map((corridor, i) => {
        const pathCoords = corridor.path.map(p => [p.lat, p.lon]);
        const corridorColor = getColor(corridor.safety_score);
        const corridorWidth = getCorridorWidth(corridor.usage_count);
        const corridorOpacity = getCorridorOpacity(corridor.safety_score);
        const isHighDanger = corridor.safety_score < 30;
        
        return (
          <>
            {/* Shadow/Glow effect for high danger corridors */}
            {isHighDanger && (
              <Polyline
                key={`corridor-glow-${i}`}
                positions={pathCoords}
                pathOptions={{
                  color: corridorColor,
                  weight: corridorWidth + 4,
                  opacity: 0.2,
                  dashArray: "10, 10",
                  lineCap: "round",
                  lineJoin: "round"
                }}
              />
            )}
            {/* Main corridor line */}
            <Polyline
              key={`corridor-${i}`}
              positions={pathCoords}
              pathOptions={{
                color: corridorColor,
                weight: corridorWidth,
                opacity: corridorOpacity,
                dashArray: "10, 8",
                lineCap: "round",
                lineJoin: "round"
              }}
              eventHandlers={{
                mouseover: (e) => {
                  e.target.setStyle({ weight: corridorWidth + 2, opacity: 1 });
                },
                mouseout: (e) => {
                  e.target.setStyle({ weight: corridorWidth, opacity: corridorOpacity });
                }
              }}
            >
              <Popup>
                <div style={{ minWidth: "220px" }}>
                  <div style={{ 
                    borderBottom: "2px solid " + corridorColor, 
                    paddingBottom: "8px", 
                    marginBottom: "8px",
                    fontWeight: "600",
                    fontSize: "14px",
                    color: "#333"
                  }}>
                    🐘 Elephant Corridor
                  </div>
                  <div style={{ fontSize: "12px", lineHeight: "1.6", color: "#555" }}>
                    <div style={{ marginBottom: "6px" }}>
                      <strong>ID:</strong> {corridor.corridor_id}
                    </div>
                    <div style={{ marginBottom: "6px" }}>
                      <strong>Route:</strong> Node {corridor.from_node} → {corridor.to_node}
                    </div>
                    <div style={{ marginBottom: "6px" }}>
                      <strong>Distance:</strong> {(corridor.distance_meters / 1000).toFixed(2)} km
                    </div>
                    <div style={{ marginBottom: "6px", display: "flex", justifyContent: "space-between" }}>
                      <span><strong>Usage:</strong> {corridor.usage_count}×</span>
                      <span><strong>Crossings:</strong> {corridor.crossing_count}</span>
                    </div>
                    <div style={{ marginBottom: "6px" }}>
                      <strong>Active Hours:</strong> {corridor.active_hours.map(h => h + ':00').join(", ")}
                    </div>
                    <div style={{ marginBottom: "6px" }}>
                      <strong>Human Distance:</strong> {(corridor.avg_human_distance / 1000).toFixed(2)} km
                    </div>
                    <div style={{ 
                      marginTop: "8px", 
                      padding: "6px", 
                      backgroundColor: corridorColor + "20",
                      borderRadius: "4px",
                      textAlign: "center",
                      fontWeight: "600"
                    }}>
                      Safety Score: {corridor.safety_score.toFixed(1)}
                    </div>
                  </div>
                </div>
              </Popup>
            </Polyline>
          </>
        );
      })}
      {filteredNodes.map((n, i) => {
        const nodeColor = getColor(n.safety_score);
        return (
          <div key={i}>
            {/* Outer glow circle */}
            <Circle
              center={[n.center_lat, n.center_lon]}
              radius={n.radius_meters + 50}
              pathOptions={{ 
                color: nodeColor, 
                weight: 0,
                fillColor: nodeColor,
                fillOpacity: 0.1
              }}
            />
            {/* Main circle */}
            <Circle
              center={[n.center_lat, n.center_lon]}
              radius={n.radius_meters}
              pathOptions={{ 
                color: nodeColor,
                weight: 2,
                fillColor: nodeColor,
                fillOpacity: getOpacity(n.elephant_count)
              }}
            />

            {zoom >= 10 && (
              <Marker position={[n.center_lat, n.center_lon]} icon={elephantIcon}>
                <Popup>
                  <div style={{ minWidth: "200px" }}>
                    <div style={{ 
                      borderBottom: "2px solid " + nodeColor, 
                      paddingBottom: "8px", 
                      marginBottom: "8px",
                      fontWeight: "600",
                      fontSize: "14px",
                      color: "#333"
                    }}>
                      🐘 Elephant Hotspot
                    </div>
                    <div style={{ fontSize: "12px", lineHeight: "1.6", color: "#555" }}>
                      <div style={{ marginBottom: "6px" }}>
                        <strong>District:</strong> {n.district}
                      </div>
                      <div style={{ marginBottom: "6px" }}>
                        <strong>Node ID:</strong> {n.node_id}
                      </div>
                      <div style={{ marginBottom: "6px", display: "flex", justifyContent: "space-between" }}>
                        <span><strong>Elephants:</strong> {n.elephant_count}</span>
                        <span><strong>Sightings:</strong> {n.sighting_count}</span>
                      </div>
                      <div style={{ marginBottom: "6px" }}>
                        <strong>Active Hours:</strong> {n.active_hours.map(h => h + ':00').join(", ")}
                      </div>
                      <div style={{ marginBottom: "6px" }}>
                        <strong>NDVI:</strong> {n.avg_ndvi.toFixed(3)}
                      </div>
                      <div style={{ marginBottom: "6px" }}>
                        <strong>Human Distance:</strong> {(n.avg_human_distance / 1000).toFixed(2)} km
                      </div>
                      <div style={{ 
                        marginTop: "8px", 
                        padding: "6px", 
                        backgroundColor: nodeColor + "20",
                        borderRadius: "4px",
                        textAlign: "center",
                        fontWeight: "600"
                      }}>
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
