import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Tooltip,
  Popup,
} from "react-leaflet";
import Papa from "papaparse";
import { useEffect, useMemo, useState } from "react";
import "leaflet/dist/leaflet.css";
import "./HimashiRiskMap.css";

/** ─── Helpers ─────────────────────────────────────────── */

function norm(str) {
  return String(str ?? "").trim().toLowerCase();
}

function getColor(level) {
  const L = norm(level);
  if (L === "high") return "#ef4444";
  if (L === "medium") return "#f59e0b";
  return "#22c55e";
}

// Handles text labels AND numeric encoding: "1" = train, "0" = road
function getVehicleType(p) {
  const v = norm(p.vehicle_type ?? p.vehicle ?? p.transport ?? p.mode ?? "");
  if (v === "1" || v.includes("train") || v.includes("rail")) return "TRAIN";
  if (v === "0" || v.includes("road") || v.includes("vehicle") || v.includes("car")) return "ROAD";
  return "UNKNOWN";
}

// Tries several field name variants, then falls back to risk_score threshold
function getRiskLevel(p) {
  const raw = norm(
    p.risk_level ?? p.Risk_Level ?? p.RiskLevel ?? p.RISK_LEVEL ?? ""
  );
  if (raw === "high" || raw === "medium" || raw === "low") return raw;
  // Numeric fallback
  const score = Number(p.risk_score ?? p.RiskScore ?? p.risk_score_norm ?? "");
  if (!Number.isNaN(score)) {
    if (score >= 0.7) return "high";
    if (score >= 0.4) return "medium";
    return "low";
  }
  return "low";
}

function getDistrict(p) {
  return String(
    p.district ?? p.District ?? p.admin_district ?? p.location_district ?? ""
  ).trim();
}

export default function HimashiRiskMap() {
  const [points, setPoints] = useState([]);
  const [basemap, setBasemap] = useState("normal");
  const [showTrain, setShowTrain] = useState(true);
  const [showRoad, setShowRoad] = useState(true);
  const [showClusters, setShowClusters] = useState(true);
  const [onlyHigh, setOnlyHigh] = useState(false);

  useEffect(() => {
    const parseAndStore = (text) => {
      const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
      const rows = parsed.data ?? [];

      // Deduplicate by lat+lon so overlapping points don't count twice on the map
      const seen = new Set();
      const unique = rows.filter((p) => {
        const lat = Number(p.latitude ?? p.lat);
        const lon = Number(p.longitude ?? p.lng ?? p.lon);
        if (Number.isNaN(lat) || Number.isNaN(lon)) return false;
        const key = `${lat.toFixed(5)}_${lon.toFixed(5)}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });

      setPoints(unique);
    };

    fetch("/risk_map_data.csv")
      .then((res) => {
        if (!res.ok) throw new Error("risk_map_data.csv not found");
        return res.text();
      })
      .then(parseAndStore)
      .catch(() => {
        // Fallback to main collision dataset
        fetch("/collision.csv")
          .then((res) => res.text())
          .then(parseAndStore)
          .catch((err) => console.error("CSV load error", err));
      });
  }, []);

  const baseLayers = {
    normal: {
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      attribution: "© OpenStreetMap",
    },
    satellite: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: "© Esri",
    },
    dark: {
      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      attribution: "© CartoDB",
    },
  };

  const filteredPoints = useMemo(() => {
    return points.filter((p) => {
      const lat = Number(p.latitude ?? p.lat);
      const lon = Number(p.longitude ?? p.lng ?? p.lon);
      if (Number.isNaN(lat) || Number.isNaN(lon)) return false;

      const vt = getVehicleType(p);
      if (vt === "TRAIN" && !showTrain) return false;
      if (vt === "ROAD" && !showRoad) return false;
      if (vt === "UNKNOWN" && !(showTrain || showRoad)) return false;

      if (onlyHigh) {
        if (getRiskLevel(p) !== "high") return false;
      }

      return true;
    });
  }, [points, showTrain, showRoad, onlyHigh]);

  const clusterCenters = useMemo(() => {
    const clusters = {};

    filteredPoints.forEach((p) => {
      const clusterId = String(p.cluster_id ?? p.cluster ?? "").trim();
      if (!clusterId || clusterId === "-1") return;

      const lat = Number(p.latitude ?? p.lat);
      const lon = Number(p.longitude ?? p.lng ?? p.lon);
      if (Number.isNaN(lat) || Number.isNaN(lon)) return;

      if (!clusters[clusterId]) {
        clusters[clusterId] = { latSum: 0, lonSum: 0, count: 0, maxRisk: 0 };
      }

      clusters[clusterId].latSum += lat;
      clusters[clusterId].lonSum += lon;
      clusters[clusterId].count += 1;

      const rs = Number(p.risk_score);
      clusters[clusterId].maxRisk = Math.max(
        clusters[clusterId].maxRisk,
        Number.isNaN(rs) ? 0 : rs
      );
    });

    return Object.entries(clusters).map(([id, c]) => ({
      cluster_id: id,
      lat: c.latSum / c.count,
      lon: c.lonSum / c.count,
      count: c.count,
      maxRisk: c.maxRisk,
    }));
  }, [filteredPoints]);

  const topDistricts = useMemo(() => {
    const map = new Map();

    filteredPoints.forEach((p) => {
      const d = getDistrict(p) || "Unknown";
      const isHigh = getRiskLevel(p) === "high";
      map.set(d, (map.get(d) ?? 0) + (isHigh ? 1 : 0));
    });

    const allZero = Array.from(map.values()).every((v) => v === 0);
    if (allZero) {
      map.clear();
      filteredPoints.forEach((p) => {
        const d = getDistrict(p) || "Unknown";
        map.set(d, (map.get(d) ?? 0) + 1);
      });
    }

    return Array.from(map.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
  }, [filteredPoints]);

  const counts = useMemo(() => {
    let train = 0, road = 0, high = 0, medium = 0, low = 0;
    filteredPoints.forEach((p) => {
      const vt = getVehicleType(p);
      if (vt === "TRAIN") train += 1;
      else if (vt === "ROAD") road += 1;

      const rl = getRiskLevel(p);
      if (rl === "high") high += 1;
      else if (rl === "medium") medium += 1;
      else low += 1;
    });
    return { total: filteredPoints.length, train, road, high, medium, low };
  }, [filteredPoints]);

  return (
    <div className="rm-page">
      {/* Control Panel */}
      <div className="rm-panel">
        <div className="rm-panel__title">
          <div>
            <div className="rm-h">Risk Map Controls</div>
            <div className="rm-sub">Filter incidents &amp; switch basemap</div>
          </div>
          <div className="rm-badge">{counts.total} points</div>
        </div>

        {/* Basemap */}
        <div className="rm-section">
          <div className="rm-label">Basemap</div>
          <div className="rm-segment">
            {["normal", "satellite", "dark"].map((t) => (
              <button
                key={t}
                className={`rm-segbtn ${basemap === t ? "is-active" : ""}`}
                onClick={() => setBasemap(t)}
                type="button"
              >
                {t === "normal" ? "Normal" : t === "satellite" ? "Satellite" : "Dark"}
              </button>
            ))}
          </div>
        </div>

        {/* Risk Analysis */}
        <div className="rm-section">
          <div className="rm-label">Risk Analysis</div>
          <div className="rm-stat-grid">
            <div className="rm-stat-card is-high">
              <div className="rm-stat-label">High Risk</div>
              <div className="rm-stat-value">{counts.high}</div>
            </div>
            <div className="rm-stat-card is-med">
              <div className="rm-stat-label">Medium</div>
              <div className="rm-stat-value">{counts.medium}</div>
            </div>
            <div className="rm-stat-card is-low">
              <div className="rm-stat-label">Low Risk</div>
              <div className="rm-stat-value">{counts.low}</div>
            </div>
            <div className="rm-stat-card is-cluster">
              <div className="rm-stat-label">Clusters</div>
              <div className="rm-stat-value">{clusterCenters.length}</div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="rm-section">
          <div className="rm-label">Vehicle Type</div>
          <label className="rm-check">
            <input type="checkbox" checked={showTrain} onChange={(e) => setShowTrain(e.target.checked)} />
            <span className="rm-check__text">Train <span className="rm-mini">({counts.train})</span></span>
          </label>
          <label className="rm-check">
            <input type="checkbox" checked={showRoad} onChange={(e) => setShowRoad(e.target.checked)} />
            <span className="rm-check__text">Road <span className="rm-mini">({counts.road})</span></span>
          </label>
          <label className="rm-check">
            <input type="checkbox" checked={onlyHigh} onChange={(e) => setOnlyHigh(e.target.checked)} />
            <span className="rm-check__text">Only High Risk</span>
          </label>
          <label className="rm-check">
            <input type="checkbox" checked={showClusters} onChange={(e) => setShowClusters(e.target.checked)} />
            <span className="rm-check__text">Show Cluster Centers</span>
          </label>
        </div>

        {/* Legend */}
        <div className="rm-section">
          <div className="rm-label">Legend</div>
          <div className="rm-legend">
            <div className="rm-legend__row"><span className="rm-dot is-high" /> High Risk</div>
            <div className="rm-legend__row"><span className="rm-dot is-med" /> Medium Risk</div>
            <div className="rm-legend__row"><span className="rm-dot is-low" /> Low Risk</div>
            <div className="rm-legend__row"><span className="rm-dot is-cluster" /> Cluster Center</div>
          </div>
        </div>

        {/* Top districts */}
        <div className="rm-section">
          <div className="rm-label">Top 5 Dangerous Districts</div>
          <div className="rm-table">
            {topDistricts.length === 0 ? (
              <div className="rm-muted">No district data found in CSV.</div>
            ) : (
              topDistricts.map(([d, c], idx) => (
                <div key={d + idx} className="rm-row">
                  <div className="rm-rank">{idx + 1}</div>
                  <div className="rm-district">{d}</div>
                  <div className="rm-count">{c}</div>
                </div>
              ))
            )}
          </div>
          <div className="rm-footnote">
            *Computed from High-risk points (fallback: total points if risk labels missing)
          </div>
        </div>
      </div>

      {/* Map */}
      <MapContainer center={[7.8731, 80.7718]} zoom={7} className="rm-map">
        <TileLayer
          url={baseLayers[basemap].url}
          attribution={baseLayers[basemap].attribution}
        />

        {filteredPoints.map((p, i) => {
          const lat = Number(p.latitude ?? p.lat);
          const lon = Number(p.longitude ?? p.lng ?? p.lon);
          if (Number.isNaN(lat) || Number.isNaN(lon)) return null;

          const vt = getVehicleType(p);
          const vtLabel = vt === "TRAIN" ? "Train" : vt === "ROAD" ? "Road" : "Unknown";
          const rl = getRiskLevel(p);
          const animal = String(p.animal_type ?? p.Animal_Type ?? p.animal ?? "").trim() || "Unknown";

          return (
            <CircleMarker
              key={`pt-${i}`}
              center={[lat, lon]}
              radius={rl === "high" ? 8 : rl === "medium" ? 6 : 5}
              pathOptions={{ color: getColor(rl), fillOpacity: 0.75 }}
            >
              <Tooltip direction="top" offset={[0, -6]} opacity={1}>
                <div style={{ fontSize: 12 }}>
                  <b>{vtLabel} Incident</b><br />
                  Risk: <b>{rl.toUpperCase()}</b><br />
                  Lat: {lat.toFixed(5)}<br />
                  Lon: {lon.toFixed(5)}
                </div>
              </Tooltip>
              <Popup>
                <div style={{ fontSize: 13 }}>
                  <b>Vehicle:</b> {vtLabel}<br />
                  <b>Animal:</b> {animal}<br />
                  <b>District:</b> {getDistrict(p) || "Unknown"}<br />
                  <b>Risk Level:</b> {rl.charAt(0).toUpperCase() + rl.slice(1)}<br />
                  <b>Risk Score:</b>{" "}
                  {Number.isNaN(Number(p.risk_score)) ? "N/A" : Number(p.risk_score).toFixed(2)}<br />
                  <b>Cluster:</b> {String(p.cluster_id ?? "N/A")}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        {showClusters &&
          clusterCenters.map((c) => (
            <CircleMarker
              key={`cluster-${c.cluster_id}`}
              center={[c.lat, c.lon]}
              radius={Math.min(28, 10 + c.count)}
              pathOptions={{ color: "#7c3aed", fillColor: "#7c3aed", fillOpacity: 0.25 }}
            >
              <Popup>
                <div style={{ fontSize: 13 }}>
                  <b>Cluster ID:</b> {c.cluster_id}<br />
                  <b>Incidents:</b> {c.count}<br />
                  <b>Max Risk:</b> {c.maxRisk.toFixed(2)}
                </div>
              </Popup>
            </CircleMarker>
          ))}
      </MapContainer>
    </div>
  );
}
