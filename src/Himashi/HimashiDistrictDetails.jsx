import { useParams, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { MapContainer, TileLayer, CircleMarker, Popup, Tooltip as LeafletTooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import Sidebar from "./HimashiSidebar";
import "./HimashiDashboard.css";

export default function DistrictDetails() {
  const { district: districtParam } = useParams();
  const navigate = useNavigate();
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  const district = decodeURIComponent(districtParam || "");

  // Load CSV
  useEffect(() => {
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
      .finally(() => setLoading(false));
  }, []);

  // Filter data for selected district
  const districtData = useMemo(() => {
    return rows.filter((r) => {
      const d = (r.District || "").trim();
      return d === district;
    });
  }, [rows, district]);

  // Calculate statistics
  const stats = useMemo(() => {
    const total = districtData.length;

    // Vehicle type breakdown
    const vehicleTypeMap = districtData.reduce((acc, r) => {
      const vt = (r.vehicle_type || "").toLowerCase().trim();
      if (!vt) return acc;
      acc[vt] = (acc[vt] || 0) + 1;
      return acc;
    }, {});

    const vehicleTypeArray = Object.entries(vehicleTypeMap)
      .map(([type, count]) => ({ name: type.charAt(0).toUpperCase() + type.slice(1), value: count }))
      .sort((a, b) => b.value - a.value);

    // Animal type breakdown
    const animalTypeMap = districtData.reduce((acc, r) => {
      const at = (r.animal_type || "").toLowerCase().trim();
      if (!at) return acc;
      acc[at] = (acc[at] || 0) + 1;
      return acc;
    }, {});

    const animalTypeArray = Object.entries(animalTypeMap)
      .map(([type, count]) => ({ name: type.charAt(0).toUpperCase() + type.slice(1), value: count }))
      .sort((a, b) => b.value - a.value);

    // Train collisions
    const trainCollisions = districtData.filter(
      (r) => (r.vehicle_type || "").toLowerCase() === "train"
    ).length;

    // Regional breakdown
    const regionMap = districtData.reduce((acc, r) => {
      const region = (r.Region || "").trim();
      if (!region) return acc;
      acc[region] = (acc[region] || 0) + 1;
      return acc;
    }, {});

    const regionArray = Object.entries(regionMap)
      .map(([region, count]) => ({ name: region, value: count }))
      .sort((a, b) => b.value - a.value);

    // Division breakdown
    const divisionMap = districtData.reduce((acc, r) => {
      const division = (r.DS_Division || "").trim();
      if (!division) return acc;
      acc[division] = (acc[division] || 0) + 1;
      return acc;
    }, {});

    const divisionArray = Object.entries(divisionMap)
      .map(([division, count]) => ({ name: division, value: count }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);

    // Coordinates for map
    const coordinates = districtData
      .filter((r) => {
        const lat = Number(r.latitude);
        const lon = Number(r.longitude);
        return !Number.isNaN(lat) && !Number.isNaN(lon);
      })
      .map((r) => ({
        lat: Number(r.latitude),
        lon: Number(r.longitude),
        animalType: r.animal_type,
        vehicleType: r.vehicle_type,
        region: r.Region,
        division: r.DS_Division,
      }));

    // Calculate center for map
    let mapCenter = [7.8731, 80.7718]; // Default Sri Lanka center
    if (coordinates.length > 0) {
      const avgLat = coordinates.reduce((sum, c) => sum + c.lat, 0) / coordinates.length;
      const avgLon = coordinates.reduce((sum, c) => sum + c.lon, 0) / coordinates.length;
      mapCenter = [avgLat, avgLon];
    }

    return {
      total,
      trainCollisions,
      vehicleTypeArray,
      animalTypeArray,
      regionArray,
      divisionArray,
      vehicleTypeCount: Object.keys(vehicleTypeMap).length,
      animalTypeCount: Object.keys(animalTypeMap).length,
      coordinates,
      mapCenter,
    };
  }, [districtData]);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF7C7C'];

  if (loading) {
    return (
      <div className="dash">
        <Sidebar />
        <main className="dash__content">
          <div style={{ padding: '40px', textAlign: 'center' }}>
            <p>Loading district data...</p>
          </div>
        </main>
      </div>
    );
  }

  if (districtData.length === 0) {
    return (
      <div className="dash">
        <Sidebar />
        <main className="dash__content">
          <div className="dash__header">
            <div>
              <h2 className="dash__title">District Details</h2>
            </div>
            <button className="dash__refresh" onClick={() => navigate(-1)}>
              ← Back
            </button>
          </div>
          <div style={{ padding: '40px', textAlign: 'center', color: '#999' }}>
            <h3>No data found for {district}</h3>
            <p>The selected district has no collision records in the dataset.</p>
            <button
              onClick={() => navigate('/dashboard')}
              style={{
                marginTop: '20px',
                padding: '10px 20px',
                backgroundColor: '#0088FE',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Go to Dashboard
            </button>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="dash">
      <Sidebar />
      <main className="dash__content">
        <div className="dash__header">
          <div>
            <h2 className="dash__title">{district} - District Details</h2>
            <p className="dash__subtitle">
              Accident statistics and analysis for {district}
            </p>
          </div>

          <div className="dash__badge">
            {stats.total} collisions
          </div>

          <button className="dash__refresh" onClick={() => navigate(-1)}>
            ← Back
          </button>
        </div>

        {/* KPI CARDS */}
        <section className="dash__kpis">
          <KpiCard
            title="Total Collisions"
            value={stats.total}
            hint="In this district"
          />
          <KpiCard
            title="Train Collisions"
            value={stats.trainCollisions}
            hint={stats.total > 0 ? `${((stats.trainCollisions / stats.total) * 100).toFixed(1)}% of total` : "—"}
          />
          <KpiCard
            title="Vehicle Types"
            value={stats.vehicleTypeCount}
            hint="Types involved"
          />
          <KpiCard
            title="Animal Species"
            value={stats.animalTypeCount}
            hint="Species involved"
          />
        </section>

        {/* PANELS */}
        <section className="dash__grid">
          {/* Vehicle Type Breakdown */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Collisions by Vehicle Type</h3>
              <span className="panel__meta">District Analysis</span>
            </div>

            {stats.vehicleTypeArray.length === 0 ? (
              <p className="panel__note">No vehicle data available.</p>
            ) : (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stats.vehicleTypeArray}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#0088FE" radius={[8, 8, 0, 0]}>
                    {stats.vehicleTypeArray.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Animal Type Breakdown */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Collisions by Animal Type</h3>
              <span className="panel__meta">Species involved</span>
            </div>

            {stats.animalTypeArray.length === 0 ? (
              <p className="panel__note">No animal data available.</p>
            ) : (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stats.animalTypeArray}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#00C49F" radius={[8, 8, 0, 0]}>
                    {stats.animalTypeArray.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Region Breakdown */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Collisions by Region</h3>
              <span className="panel__meta">Geographic breakdown</span>
            </div>

            {stats.regionArray.length === 0 ? (
              <p className="panel__note">No region data available.</p>
            ) : (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stats.regionArray}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#FF8042" radius={[8, 8, 0, 0]}>
                    {stats.regionArray.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Division Breakdown */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Top Divisions in {district}</h3>
              <span className="panel__meta">DS Division breakdown</span>
            </div>

            {stats.divisionArray.length === 0 ? (
              <p className="panel__note">No division data available.</p>
            ) : (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stats.divisionArray}>
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#FFBB28" radius={[8, 8, 0, 0]}>
                    {stats.divisionArray.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Collision Hotspots Map */}
          <div className="panel" style={{ gridColumn: '1 / -1' }}>
            <div className="panel__head">
              <h3 className="panel__title">Collision Hotspots Map</h3>
              <span className="panel__meta">Geographic distribution</span>
            </div>

            <div style={{ height: '400px', borderRadius: '8px', overflow: 'hidden' }}>
              <MapContainer
                center={stats.mapCenter}
                zoom={10}
                style={{ height: '100%', width: '100%' }}
              >
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                {stats.coordinates.map((coord, idx) => (
                  <CircleMarker
                    key={`hotspot-${idx}`}
                    center={[coord.lat, coord.lon]}
                    radius={5}
                    pathOptions={{
                      color: '#0088FE',
                      fillOpacity: 0.7,
                    }}
                  >
                    <LeafletTooltip>
                      <div>
                        <b>{coord.vehicleType || 'Unknown'}</b><br/>
                        Animal: {coord.animalType || 'Unknown'}<br/>
                        Division: {coord.division || 'Unknown'}
                      </div>
                    </LeafletTooltip>
                  </CircleMarker>
                ))}
              </MapContainer>
            </div>

            <p className="panel__note">
              All collision points within {district}. Blue markers show incident locations.
            </p>
          </div>

          {/* Statistics Summary */}
          <div className="panel" style={{ gridColumn: '1 / -1' }}>
            <div className="panel__head">
              <h3 className="panel__title">Statistics Summary</h3>
              <span className="panel__meta">Key metrics for {district}</span>
            </div>

            <div style={{ padding: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px' }}>
                <StatItem label="Total Collisions" value={stats.total} />
                <StatItem label="Train Collisions" value={stats.trainCollisions} />
                <StatItem label="Road Collisions" value={stats.total - stats.trainCollisions} />
                <StatItem label="Vehicle Types" value={stats.vehicleTypeCount} />
                <StatItem label="Animal Species" value={stats.animalTypeCount} />
                <StatItem label="Regions" value={stats.regionArray.length} />
                <StatItem label="Divisions" value={stats.divisionArray.length} />
                <StatItem
                  label="Train %"
                  value={stats.total > 0 ? `${((stats.trainCollisions / stats.total) * 100).toFixed(1)}%` : '—'}
                />
              </div>
            </div>

            <p className="panel__note">
              All data dynamically calculated from the collision dataset for {district}.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

/* ============ Helper Components ============ */

function KpiCard({ title, value, hint }) {
  return (
    <div className="kpi">
      <div className="kpi__title">{title}</div>
      <div className="kpi__value">{value}</div>
      <div className="kpi__hint">{hint}</div>
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
      <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px', color: '#333' }}>
        {value}
      </div>
    </div>
  );
}
