import { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import 'leaflet/dist/leaflet.css';
import Sidebar from "./HimashiSidebar";
import "./HimashiDashboard.css";

export default function TopDangerousDistricts() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

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

  // Calculate all districts with statistics
  const districtsStats = useMemo(() => {
    const districtMap = {};

    rows.forEach((r) => {
      const district = (r.District || "").trim();
      if (!district) return;

      if (!districtMap[district]) {
        districtMap[district] = {
          totalCollisions: 0,
          trainCollisions: 0,
          roadCollisions: 0,
          vehicleTypes: new Set(),
          animalTypes: new Set(),
          regions: new Set(),
          coordinates: [],
        };
      }

      const stats = districtMap[district];
      stats.totalCollisions += 1;

      const vt = (r.vehicle_type || "").toLowerCase();
      if (vt === "train") {
        stats.trainCollisions += 1;
      } else if (vt.includes("car") || vt.includes("bus") || vt.includes("vehicle")) {
        stats.roadCollisions += 1;
      }

      const vehicleType = r.vehicle_type || "Unknown";
      stats.vehicleTypes.add(vehicleType);

      const animalType = r.animal_type || "Unknown";
      stats.animalTypes.add(animalType);

      const region = r.Region || "Unknown";
      stats.regions.add(region);

      const lat = Number(r.latitude);
      const lon = Number(r.longitude);
      if (!Number.isNaN(lat) && !Number.isNaN(lon)) {
        stats.coordinates.push({ lat, lon });
      }
    });

    return Object.entries(districtMap)
      .map(([district, stats]) => ({
        district,
        totalCollisions: stats.totalCollisions,
        trainCollisions: stats.trainCollisions,
        roadCollisions: stats.roadCollisions,
        vehicleTypeCount: stats.vehicleTypes.size,
        animalTypeCount: stats.animalTypes.size,
        regionCount: stats.regions.size,
        coordinateCount: stats.coordinates.length,
      }))
      .sort((a, b) => b.totalCollisions - a.totalCollisions);
  }, [rows]);

  // Top 10 dangerous districts
  const topDistricts = useMemo(() => {
    return districtsStats.slice(0, 10);
  }, [districtsStats]);

  const COLORS = ['#ef4444', '#ff6b6b', '#ff8787', '#ff9999', '#ff9999', '#ffb3b3', '#ffcccc', '#ffe0e0', '#fff0f0'];

  const handleDistrictClick = (district) => {
    navigate(`/district/${encodeURIComponent(district)}`);
  };

  if (loading) {
    return (
      <div className="dash">
        <Sidebar />
        <main className="dash__content">
          <div style={{ padding: '40px', textAlign: 'center' }}>
            <p>Loading dangerous districts...</p>
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
            <h2 className="dash__title">Top Dangerous Districts</h2>
            <p className="dash__subtitle">
              Districts with the highest collision counts from the dataset
            </p>
          </div>

          <div className="dash__badge">
            {districtsStats.length} districts analyzed
          </div>
        </div>

        {/* KPI CARDS */}
        <section className="dash__kpis">
          <KpiCard
            title="Total Districts"
            value={districtsStats.length}
            hint="In dataset"
          />
          <KpiCard
            title="Most Dangerous"
            value={topDistricts[0]?.district || "—"}
            hint={topDistricts[0] ? `${topDistricts[0].totalCollisions} collisions` : "—"}
          />
          <KpiCard
            title="Total Collisions"
            value={rows.length}
            hint="Across all districts"
          />
          <KpiCard
            title="Avg per District"
            value={districtsStats.length > 0 ? Math.round(rows.length / districtsStats.length) : 0}
            hint="Average collisions"
          />
        </section>

        {/* PANELS */}
        <section className="dash__grid">
          {/* Top 10 Bar Chart */}
          <div className="panel" style={{ gridColumn: '1 / -1' }}>
            <div className="panel__head">
              <h3 className="panel__title">Top 10 Most Dangerous Districts</h3>
              <span className="panel__meta">By collision count</span>
            </div>

            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={topDistricts}>
                <XAxis
                  dataKey="district"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                />
                <YAxis />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                    padding: '8px'
                  }}
                  cursor={{ fill: 'rgba(0,0,0,0.1)' }}
                />
                <Bar
                  dataKey="totalCollisions"
                  fill="#ef4444"
                  radius={[8, 8, 0, 0]}
                  onClick={(data) => handleDistrictClick(data.district)}
                  style={{ cursor: 'pointer' }}
                >
                  {topDistricts.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            <p className="panel__note">
              Click on any bar to view detailed statistics for that district.
            </p>
          </div>

          {/* Detailed Ranking Table */}
          <div className="panel" style={{ gridColumn: '1 / -1' }}>
            <div className="panel__head">
              <h3 className="panel__title">Detailed Ranking - Top 20 Districts</h3>
              <span className="panel__meta">Comprehensive analysis</span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f5f5f5', borderBottom: '2px solid #0088FE' }}>
                    <th style={{ padding: '12px', textAlign: 'left', fontWeight: 'bold' }}>Rank</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontWeight: 'bold' }}>District</th>
                    <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>Total Collisions</th>
                    <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>Train</th>
                    <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>Road</th>
                    <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>Vehicle Types</th>
                    <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>Animal Types</th>
                    <th style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {districtsStats.slice(0, 20).map((d, idx) => (
                    <tr
                      key={d.district}
                      style={{
                        borderBottom: '1px solid #eee',
                        backgroundColor: idx % 2 === 0 ? '#fff' : '#f9f9f9',
                        cursor: 'pointer',
                        transition: 'background-color 0.2s'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f0f0f0'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = idx % 2 === 0 ? '#fff' : '#f9f9f9'}
                    >
                      <td style={{ padding: '12px', fontWeight: 'bold', color: '#0088FE' }}>#{idx + 1}</td>
                      <td style={{ padding: '12px' }}>
                        <strong>{d.district}</strong>
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right', color: '#ef4444', fontWeight: 'bold' }}>
                        {d.totalCollisions}
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>{d.trainCollisions}</td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>{d.roadCollisions}</td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>{d.vehicleTypeCount}</td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>{d.animalTypeCount}</td>
                      <td style={{ padding: '12px', textAlign: 'center' }}>
                        <button
                          onClick={() => handleDistrictClick(d.district)}
                          style={{
                            padding: '6px 12px',
                            backgroundColor: '#0088FE',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '12px'
                          }}
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="panel__note">
              All statistics are dynamically calculated from the collision dataset. Click "View" or district name to see detailed analysis.
            </p>
          </div>

          {/* Summary Cards */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Summary Statistics</h3>
              <span className="panel__meta">Overview</span>
            </div>

            <div style={{ padding: '16px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <StatItem
                label="Most Dangerous"
                value={topDistricts[0]?.district || "—"}
                subtext={topDistricts[0] ? `${topDistricts[0].totalCollisions} collisions` : ""}
              />
              <StatItem
                label="2nd Most Dangerous"
                value={topDistricts[1]?.district || "—"}
                subtext={topDistricts[1] ? `${topDistricts[1].totalCollisions} collisions` : ""}
              />
              <StatItem
                label="3rd Most Dangerous"
                value={topDistricts[2]?.district || "—"}
                subtext={topDistricts[2] ? `${topDistricts[2].totalCollisions} collisions` : ""}
              />
              <StatItem
                label="Avg Collisions"
                value={districtsStats.length > 0 ? (rows.length / districtsStats.length).toFixed(1) : "—"}
                subtext="per district"
              />
            </div>
          </div>

          {/* Collision Distribution */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Collision Distribution</h3>
              <span className="panel__meta">Key insights</span>
            </div>

            <div style={{ padding: '16px' }}>
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '13px', color: '#666', marginBottom: '4px' }}>Total Collisions</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ef4444' }}>{rows.length}</div>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '13px', color: '#666', marginBottom: '4px' }}>Train Collisions</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#0088FE' }}>
                  {rows.filter(r => (r.vehicle_type || "").toLowerCase() === "train").length}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '13px', color: '#666', marginBottom: '4px' }}>Districts Analyzed</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#00C49F' }}>{districtsStats.length}</div>
              </div>
            </div>
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

function StatItem({ label, value, subtext }) {
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
      {subtext && <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>{subtext}</div>}
    </div>
  );
}
