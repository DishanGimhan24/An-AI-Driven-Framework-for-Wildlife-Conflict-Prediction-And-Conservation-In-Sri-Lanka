import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, Cell, CartesianGrid,
} from "recharts";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import Sidebar from "./HimashiSidebar";
import "./HimashiDashboard.css";

/* ── Palette ────────────────────────────────────────────────── */
const PALETTE = [
  "#4ade80", "#22c55e", "#86efac", "#16a34a",
  "#a3e635", "#34d399", "#6ee7b7", "#bbf7d0",
];

const DANGER_PALETTE = [
  "#ef4444", "#f97316", "#fb923c", "#fbbf24",
  "#facc15", "#f59e0b", "#fcd34d", "#fde68a",
];

/* ── Reusable tooltip ───────────────────────────────────────── */
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div style={{ fontWeight: 700, marginBottom: 4, color: "#e6edf3" }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.fill || p.color, fontSize: 12 }}>
          {p.name}: <strong>{p.value}</strong>
        </div>
      ))}
    </div>
  );
};

/* ── District coordinates ───────────────────────────────────── */
const DISTRICT_COORDS = {
  Colombo: [6.9271, 79.8612], Gampaha: [7.0873, 80.0144],
  Kalutara: [6.5854, 79.9607], Kandy: [7.2906, 80.6337],
  Matale: [7.4675, 80.6234], "Nuwara Eliya": [6.9497, 80.7891],
  Galle: [6.0329, 80.2168], Matara: [5.9485, 80.5353],
  Hambantota: [6.1429, 81.1212], Jaffna: [9.6615, 80.0255],
  Kilinochchi: [9.3803, 80.3761], Mannar: [8.9810, 79.9044],
  Vavuniya: [8.7514, 80.4971], Mullaitivu: [9.2671, 80.8142],
  Batticaloa: [7.7307, 81.6747], Ampara: [7.2912, 81.6747],
  Trincomalee: [8.5874, 81.2152], Kurunegala: [7.4863, 80.3647],
  Puttalam: [8.0362, 79.8282], Anuradhapura: [8.3114, 80.4037],
  Polonnaruwa: [7.9403, 81.0188], Badulla: [6.9934, 81.0550],
  Moneragala: [6.7525, 81.3514], Ratnapura: [6.7056, 80.3847],
  Kegalle: [7.2513, 80.3464],
};

/* ─────────────────────────────────────────────────────────── */
export default function Dashboard() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const navigate = useNavigate();

  const loadData = () => {
    setRefreshing(true);
    fetch("/collision.csv")
      .then((r) => r.text())
      .then((text) => {
        const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
        setRows(parsed.data || []);
      })
      .catch((err) => console.error("CSV load error:", err))
      .finally(() => { setLoading(false); setRefreshing(false); });
  };

  useEffect(() => { loadData(); }, []);

  /* ── Calculations ─────────────────────────────────────────── */
  const summary = useMemo(() => {
    const total = rows.length;

    const trainCollisions = rows.filter(
      (r) => (r.vehicle_type || "").toLowerCase() === "train"
    ).length;

    const districtCounts = rows.reduce((acc, r) => {
      const d = (r.District || "").trim();
      if (d) acc[d] = (acc[d] || 0) + 1;
      return acc;
    }, {});

    const topDistricts = Object.entries(districtCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([district, count]) => ({ district, count }));

    const allDistrictCount = Object.keys(districtCounts).length;

    const vehicleTypeMap = rows.reduce((acc, r) => {
      const vt = (r.vehicle_type || "").trim();
      if (vt) acc[vt] = (acc[vt] || 0) + 1;
      return acc;
    }, {});

    const vehicleTypeArray = Object.entries(vehicleTypeMap)
      .map(([type, count]) => ({
        name: type.charAt(0).toUpperCase() + type.slice(1),
        count,
        pct: total > 0 ? +((count / total) * 100).toFixed(1) : 0,
      }))
      .sort((a, b) => b.count - a.count);

    const animalTypeMap = rows.reduce((acc, r) => {
      const at = (r.animal_type || "").trim();
      if (at) acc[at] = (acc[at] || 0) + 1;
      return acc;
    }, {});

    const animalTypeArray = Object.entries(animalTypeMap)
      .map(([type, count]) => ({
        name: type.charAt(0).toUpperCase() + type.slice(1),
        count,
        pct: total > 0 ? +((count / total) * 100).toFixed(1) : 0,
      }))
      .sort((a, b) => b.count - a.count);

    const trainPct = total > 0 ? ((trainCollisions / total) * 100).toFixed(1) : "0";

    return {
      total,
      trainCollisions,
      trainPct,
      vehicleTypeArray,
      animalTypeArray,
      topDistricts,
      allDistrictCount,
      vehicleTypeCount: Object.keys(vehicleTypeMap).length,
      animalTypeCount: Object.keys(animalTypeMap).length,
    };
  }, [rows]);

  if (loading) {
    return (
      <div className="dash">
        <Sidebar />
        <main className="dash__content">
          <div className="dash__loading">
            <div className="dash__loading-text">Loading data…</div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="dash">
      <Sidebar />

      <main className="dash__content">
        {/* ── Header ──────────────────────────────────────── */}
        <div className="dash__header">
          <div>
            <h2 className="dash__title">Dashboard Overview</h2>
            <p className="dash__subtitle">
              Smart AVC Monitoring · Real-time wildlife collision risk analysis across Sri Lanka
            </p>
          </div>
          <div className="dash__header-actions">
            <div className="dash__badge">
              {summary.total.toLocaleString()} incidents tracked
            </div>
            <button
              className="dash__refresh"
              onClick={loadData}
              disabled={refreshing}
            >
              {refreshing ? "↻ Refreshing…" : "↻ Refresh"}
            </button>
          </div>
        </div>

        {/* ── KPI Cards ────────────────────────────────────── */}
        <section className="dash__kpis">
          <KpiCard
            icon="🎯"
            title="Total Incidents"
            value={summary.total.toLocaleString()}
            description="Collision events in dataset"
            badge="Live"
            badgeColor="#4ade80"
            trend={null}
          />
          <KpiCard
            icon="🚂"
            title="Rail Incidents"
            value={summary.trainCollisions.toLocaleString()}
            description={`${summary.trainPct}% of all incidents`}
            badge="Analyzed"
            badgeColor="#60a5fa"
            trend={`${summary.trainPct}% of total`}
          />
          <KpiCard
            icon="🗺️"
            title="Districts Covered"
            value={summary.allDistrictCount}
            description="Geographic regions with records"
            badge="Active"
            badgeColor="#facc15"
            trend={null}
          />
          <KpiCard
            icon="🐾"
            title="Animal Species"
            value={summary.animalTypeCount}
            description={`${summary.vehicleTypeCount} vehicle types analyzed`}
            badge="Complete"
            badgeColor="#a78bfa"
            trend={null}
          />
        </section>

        {/* ── Panels ───────────────────────────────────────── */}
        <section className="dash__grid">

          {/* Vehicle Type Chart */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Collisions by Vehicle Type</h3>
              <span className="panel__meta">Transport Mode</span>
            </div>
            {summary.vehicleTypeArray.length === 0 ? (
              <EmptyState label="No vehicle data" />
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={summary.vehicleTypeArray} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#8b949e", fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#8b949e", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    width={35}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    wrapperStyle={{ fontSize: 11, color: "#8b949e", paddingTop: 8 }}
                    formatter={() => "Incident count"}
                  />
                  <Bar dataKey="count" name="Incidents" radius={[6, 6, 0, 0]}>
                    {summary.vehicleTypeArray.map((_, i) => (
                      <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
            <p className="panel__note">
              Distribution across transportation modes — identifies highest-risk vehicle categories.
            </p>
          </div>

          {/* Animal Species Chart */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Wildlife Species Involved</h3>
              <span className="panel__meta">Species Impact</span>
            </div>
            {summary.animalTypeArray.length === 0 ? (
              <EmptyState label="No species data" />
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={summary.animalTypeArray} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#8b949e", fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#8b949e", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    width={35}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    wrapperStyle={{ fontSize: 11, color: "#8b949e", paddingTop: 8 }}
                    formatter={() => "Incident count"}
                  />
                  <Bar dataKey="count" name="Incidents" radius={[6, 6, 0, 0]}>
                    {summary.animalTypeArray.map((_, i) => (
                      <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
            <p className="panel__note">
              Species breakdown helps prioritize conservation efforts for most vulnerable wildlife.
            </p>
          </div>

          {/* Top Hotspot Districts */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Top 5 Collision Hotspots</h3>
              <span className="panel__meta">Click bar to drill down</span>
            </div>
            {summary.topDistricts.length === 0 ? (
              <EmptyState label="No district data" />
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart
                  data={summary.topDistricts}
                  margin={{ top: 4, right: 8, left: -10, bottom: 0 }}
                  layout="vertical"
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: "#8b949e", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="district"
                    tick={{ fill: "#e6edf3", fontSize: 12, fontWeight: 500 }}
                    axisLine={false}
                    tickLine={false}
                    width={90}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    wrapperStyle={{ fontSize: 11, color: "#8b949e", paddingTop: 8 }}
                    formatter={() => "Collisions"}
                  />
                  <Bar
                    dataKey="count"
                    name="Collisions"
                    radius={[0, 6, 6, 0]}
                    onClick={(d) => navigate(`/district/${encodeURIComponent(d.district)}`)}
                    style={{ cursor: "pointer" }}
                  >
                    {summary.topDistricts.map((_, i) => (
                      <Cell key={i} fill={DANGER_PALETTE[i % DANGER_PALETTE.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
            <p className="panel__note">
              Horizontal layout for easy label reading. Click a bar to view district drill-down.
            </p>
          </div>

          {/* Map Preview */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Geospatial Hotspot Map</h3>
              <span className="panel__meta">Sri Lanka</span>
            </div>
            <div className="map-preview" onClick={() => navigate("/risk-map")}>
              <MapContainer
                center={[7.8731, 80.7718]}
                zoom={7}
                style={{ height: "100%", width: "100%" }}
                zoomControl={false}
                dragging={false}
                scrollWheelZoom={false}
              >
                <TileLayer
                  url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                  attribution='&copy; <a href="https://carto.com/">CARTO</a>'
                />
                {summary.topDistricts.map((d) => {
                  const coord = DISTRICT_COORDS[d.district];
                  if (!coord) return null;
                  return (
                    <Marker key={d.district} position={coord}>
                      <Popup>
                        <strong>{d.district}</strong><br />
                        {d.count} collisions
                      </Popup>
                    </Marker>
                  );
                })}
              </MapContainer>
            </div>
            <p className="panel__note">
              Top 5 hotspot districts shown. Click to open the full interactive risk map.
            </p>
          </div>

          {/* System Stats */}
          <div className="panel panel--span2">
            <div className="panel__head">
              <h3 className="panel__title">System Overview</h3>
              <span className="panel__meta">Key Metrics</span>
            </div>
            <div className="stats-grid">
              <StatItem label="Total Incidents" value={summary.total.toLocaleString()} />
              <StatItem label="Rail Collisions" value={summary.trainCollisions.toLocaleString()} />
              <StatItem label="Districts Analyzed" value={summary.allDistrictCount} />
              <StatItem label="Rail Collision Rate" value={`${summary.trainPct}%`} />
              <StatItem label="Vehicle Categories" value={summary.vehicleTypeCount} />
              <StatItem label="Species Recorded" value={summary.animalTypeCount} />
            </div>
          </div>

        </section>
      </main>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/* Helper components                                           */
/* ─────────────────────────────────────────────────────────── */

function KpiCard({ icon, title, value, description, badge, badgeColor, trend }) {
  return (
    <div className="kpi">
      <div className="kpi__top">
        <div className="kpi__icon">{icon}</div>
        {badge && (
          <span
            className="kpi__badge"
            style={{
              backgroundColor: badgeColor + "18",
              color: badgeColor,
              border: `1px solid ${badgeColor}30`,
            }}
          >
            {badge}
          </span>
        )}
      </div>
      <div className="kpi__title">{title}</div>
      <div className="kpi__value">{value}</div>
      <div className="kpi__description">{description}</div>
      {trend && (
        <div className="kpi__trend kpi__trend--neutral">
          ↗ {trend}
        </div>
      )}
    </div>
  );
}

function StatItem({ label, value }) {
  return (
    <div className="stat-item">
      <div className="stat-item__label">{label}</div>
      <div className="stat-item__value">{value}</div>
    </div>
  );
}

function EmptyState({ label }) {
  return (
    <div className="empty-state">
      <div className="empty-state__icon">📭</div>
      <div>{label}</div>
    </div>
  );
}
