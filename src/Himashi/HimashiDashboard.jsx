import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import Sidebar from "./HimashiSidebar";
import "./HimashiDashboard.css";

export default function Dashboard() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const navigate = useNavigate();

  // Load CSV from collision dataset (public/collision.csv)
  const loadData = () => {
    setRefreshing(true);
    fetch("/collision.csv")
      .then((res) => res.text())
      .then((text) => {
        const parsed = Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
        });
        setRows(parsed.data || []);
      })
      .catch((err) => console.error("CSV load error:", err))
      .finally(() => {
        setLoading(false);
        setRefreshing(false);
      });
  };

  useEffect(() => {
    loadData();
  }, []);

  // ---------- KPI + Summary Calculations ----------
  const summary = useMemo(() => {
    const total = rows.length;

    // Count total train collisions
    const trainCollisions = rows.filter(
      (r) => (r.vehicle_type || "").toLowerCase() === "train"
    ).length;

    // Vehicle type breakdown - detailed counts for each vehicle type
    const vehicleTypeMap = rows.reduce((acc, r) => {
      const vt = (r.vehicle_type || "").toLowerCase().trim();
      if (!vt) return acc;
      acc[vt] = (acc[vt] || 0) + 1;
      return acc;
    }, {});

    const vehicleTypeArray = Object.entries(vehicleTypeMap)
      .map(([type, count]) => ({ name: type.charAt(0).toUpperCase() + type.slice(1), value: count }))
      .sort((a, b) => b.value - a.value);

    // District collision counts (all collisions, not just high risk)
    const districtCounts = rows.reduce((acc, r) => {
      const d = (r.District || "").trim();
      if (!d) return acc;
      acc[d] = (acc[d] || 0) + 1;
      return acc;
    }, {});

    const topDistricts = Object.entries(districtCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([district, count]) => ({ district, count }));

    // Animal type breakdown
    const animalTypeMap = rows.reduce((acc, r) => {
      const at = (r.animal_type || "").toLowerCase().trim();
      if (!at) return acc;
      acc[at] = (acc[at] || 0) + 1;
      return acc;
    }, {});

    const animalTypeArray = Object.entries(animalTypeMap)
      .map(([type, count]) => ({ name: type.charAt(0).toUpperCase() + type.slice(1), value: count }))
      .sort((a, b) => b.value - a.value);

    return {
      total,
      trainCollisions,
      vehicleTypeArray,
      animalTypeArray,
      topDistricts,
      vehicleTypeCount: Object.keys(vehicleTypeMap).length,
      animalTypeCount: Object.keys(animalTypeMap).length,
    };
  }, [rows]);

  // Chart colors
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF7C7C'];

  // District coordinates for map
  const districtCoords = {
    'Colombo': [6.9271, 79.8612],
    'Gampaha': [7.0873, 80.0144],
    'Kalutara': [6.5854, 79.9607],
    'Kandy': [7.2906, 80.6337],
    'Matale': [7.4675, 80.6234],
    'Nuwara Eliya': [6.9497, 80.7891],
    'Galle': [6.0329, 80.2168],
    'Matara': [5.9485, 80.5353],
    'Hambantota': [6.1429, 81.1212],
    'Jaffna': [9.6615, 80.0255],
    'Kilinochchi': [9.3803, 80.3761],
    'Mannar': [8.9810, 79.9044],
    'Vavuniya': [8.7514, 80.4971],
    'Mullaitivu': [9.2671, 80.8142],
    'Batticaloa': [7.7307, 81.6747],
    'Ampara': [7.2912, 81.6747],
    'Trincomalee': [8.5874, 81.2152],
    'Kurunegala': [7.4863, 80.3647],
    'Puttalam': [8.0362, 79.8282],
    'Anuradhapura': [8.3114, 80.4037],
    'Polonnaruwa': [7.9403, 81.0188],
    'Badulla': [6.9934, 81.0550],
    'Moneragala': [6.7525, 81.3514],
    'Ratnapura': [6.7056, 80.3847],
    'Kegalle': [7.2513, 80.3464],
  };

  const handleMapClick = () => {
    navigate('/map');
  };

  const handleDistrictClick = (district) => {
    navigate(`/district/${encodeURIComponent(district)}`);
  };

  return (
    <div className="dash">
      {/* LEFT SIDEBAR */}
      <Sidebar />

      {/* MAIN CONTENT */}
      <main className="dash__content">
        <div className="dash__header">
          <div>
            <h2 className="dash__title">Dashboard Overview</h2>
            <p className="dash__subtitle">
              Smart AVC Monitoring System – Real-time collision risk analysis and hotspot identification
            </p>
          </div>

          <div className="dash__badge">
            {loading ? "Loading…" : `${summary.total} incidents tracked`}
          </div>

          <button
            className="dash__refresh"
            onClick={loadData}
            disabled={refreshing}
          >
            {refreshing ? "Refreshing…" : "🔄 Refresh"}
          </button>
        </div>

        {/* KPI CARDS */}
        <section className="dash__kpis">
          <KpiCard
            icon="🎯"
            title="Active Incidents"
            value={summary.total}
            description="Total collision events tracked"
            status="Monitoring"
          />
          <KpiCard
            icon="🚂"
            title="Rail Incidents"
            value={summary.trainCollisions}
            description={summary.total > 0 ? `${((summary.trainCollisions / summary.total) * 100).toFixed(1)}% of total incidents` : "No train data"}
            status="Analyzed"
          />
          <KpiCard
            icon="🌍"
            title="Geographic Coverage"
            value={summary.topDistricts.length}
            description="Districts with collision records"
            status="Active"
          />
          <KpiCard
            icon="📊"
            title="Analysis Ready"
            value={summary.vehicleTypeArray.length}
            description="Vehicle categories analyzed"
            status="Complete"
          />
        </section>

        {/* PANELS */}
        <section className="dash__grid">
          {/* Vehicle Type Breakdown */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Collision Risk by Vehicle Type</h3>
              <span className="panel__meta">Transportation Mode Analysis</span>
            </div>

            {summary.vehicleTypeArray.length === 0 ? (
              <p className="panel__note">No vehicle data available.</p>
            ) : (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={summary.vehicleTypeArray}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#0088FE" radius={[8, 8, 0, 0]}>
                    {summary.vehicleTypeArray.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}

            <p className="panel__note">
              Incident distribution across different transportation modes. Use this to identify which vehicle types are most at risk.
            </p>
          </div>

          {/* Animal Type Breakdown */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Wildlife Collision Species</h3>
              <span className="panel__meta">Animal Involved Assessment</span>
            </div>

            {summary.animalTypeArray.length === 0 ? (
              <p className="panel__note">No animal data available.</p>
            ) : (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={summary.animalTypeArray}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#00C49F" radius={[8, 8, 0, 0]}>
                    {summary.animalTypeArray.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}

            <p className="panel__note">
              Species breakdown of documented wildlife collisions. Helps prioritize conservation efforts for vulnerable species.
            </p>
          </div>

          {/* Top Dangerous Districts */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Collision Hotspot Zones</h3>
              <span className="panel__meta">High-Priority Intervention Areas</span>
            </div>

            {summary.topDistricts.length === 0 ? (
              <p className="panel__note">No district data available.</p>
            ) : (
              <>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={summary.topDistricts}>
                    <XAxis dataKey="district" />
                    <YAxis />
                    <Tooltip />
                    <Bar 
                      dataKey="count" 
                      fill="#FF8042" 
                      radius={[8, 8, 0, 0]}
                      onClick={(data) => handleDistrictClick(data.district)}
                      style={{ cursor: 'pointer' }}
                    />
                  </BarChart>
                </ResponsiveContainer>
                <p className="panel__note">
                  Top collision hotspots by geographic district. Click on a bar to view detailed risk analysis.
                </p>
              </>
            )}
          </div>

          {/* Interactive Map Preview */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Interactive Risk Map</h3>
              <span className="panel__meta">Geospatial Visualization</span>
            </div>

            <div 
              style={{
                height: '250px', 
                cursor: 'pointer',
                borderRadius: '8px',
                overflow: 'hidden'
              }}
              onClick={handleMapClick}
            >
              <MapContainer 
                center={[7.8731, 80.7718]} 
                zoom={7} 
                style={{ height: '100%', width: '100%' }}
              >
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                {summary.topDistricts.map((d) => {
                  const coord = districtCoords[d.district];
                  if (coord) {
                    return (
                      <Marker 
                        key={d.district} 
                        position={coord}
                        eventHandlers={{ click: () => handleDistrictClick(d.district) }}
                      >
                        <Popup>
                          <strong>{d.district}</strong><br/>
                          {d.count} collisions<br/>
                          <button 
                            style={{
                              marginTop: '8px',
                              padding: '4px 8px',
                              cursor: 'pointer',
                              backgroundColor: '#0088FE',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px'
                            }}
                            onClick={() => handleDistrictClick(d.district)}
                          >
                            View Details
                          </button>
                        </Popup>
                      </Marker>
                    );
                  }
                  return null;
                })}
              </MapContainer>
            </div>

            <p className="panel__note">
              Geographic distribution of collision hotspots. Click on the map to view the complete Risk Map with detailed incident locations and risk levels.
            </p>
          </div>

          {/* Statistics Summary Table */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">System Overview</h3>
              <span className="panel__meta">Key Performance Indicators</span>
            </div>

            <div style={{ padding: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <StatItem label="Total Incidents" value={summary.total} />
                <StatItem label="Rail Collisions" value={summary.trainCollisions} />
                <StatItem label="Data Coverage" value={summary.topDistricts.length + " districts"} />
                <StatItem 
                  label="Rail Collision Rate" 
                  value={summary.total > 0 ? `${((summary.trainCollisions / summary.total) * 100).toFixed(1)}%` : '—'} 
                />
              </div>
            </div>

            <p className="panel__note">
              Real-time metrics from the Smart AVC system. All data is dynamically calculated from the collision incident database.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

/* ============ Helper Components ============ */

function KpiCard({ icon, title, value, description, status, hint }) {
  // Support both old and new API for backward compatibility
  const displayTitle = title;
  const displayValue = value;
  const displayDescription = description || hint;
  const displayIcon = icon || "📈";
  const displayStatus = status || "Active";

  // Status color map
  const statusColorMap = {
    "Monitoring": "#0088FE",
    "Analyzed": "#00C49F",
    "Active": "#FFBB28",
    "Complete": "#22c55e",
    "Alert": "#ef4444"
  };

  const statusColor = statusColorMap[displayStatus] || "#0088FE";

  return (
    <div className="kpi" style={{
      backgroundColor: '#fff',
      borderRadius: '8px',
      padding: '24px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
      border: '1px solid #e5e7eb',
      transition: 'all 0.3s ease',
      cursor: 'pointer'
    }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = '0 4px 16px rgba(0,0,0,0.12)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
        e.currentTarget.style.transform = 'translateY(0)';
      }}
    >
      {/* Icon and Status */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
        <div style={{ fontSize: '32px' }}>{displayIcon}</div>
        <span style={{
          fontSize: '11px',
          fontWeight: '600',
          padding: '4px 10px',
          borderRadius: '20px',
          backgroundColor: statusColor + '15',
          color: statusColor,
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          {displayStatus}
        </span>
      </div>

      {/* Title */}
      <div style={{
        fontSize: '14px',
        fontWeight: '600',
        color: '#6b7280',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        marginBottom: '8px'
      }}>
        {displayTitle}
      </div>

      {/* Value */}
      <div style={{
        fontSize: '36px',
        fontWeight: '700',
        color: '#1f2937',
        marginBottom: '12px'
      }}>
        {displayValue}
      </div>

      {/* Description */}
      <div style={{
        fontSize: '13px',
        color: '#9ca3af',
        lineHeight: '1.5'
      }}>
        {displayDescription}
      </div>
    </div>
  );
}

function StatItem({ label, value }) {
  return (
    <div style={{
      padding: '12px',
      backgroundColor: '#f5f5f5',
      borderRadius: '6px',
      borderLeft: '4px solid #0088FE'
    }}>
      <div style={{ fontSize: '13px', color: '#666' }}>{label}</div>
      <div style={{ fontSize: '20px', fontWeight: 'bold', marginTop: '4px', color: '#333' }}>
        {value}
      </div>
    </div>
  );
}
