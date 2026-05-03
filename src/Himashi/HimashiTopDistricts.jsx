import { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, Cell, CartesianGrid,
} from "recharts";
import "leaflet/dist/leaflet.css";
import Sidebar from "./HimashiSidebar";
import "./HimashiDashboard.css";

const DANGER_PALETTE = [
  "#ef4444", "#f97316", "#fb923c", "#fbbf24",
  "#facc15", "#f59e0b", "#fcd34d", "#fde68a",
  "#fef08a", "#fef9c3",
];

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

export default function TopDangerousDistricts() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetch("/collision.csv")
      .then((r) => r.text())
      .then((text) => {
        const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
        setRows(parsed.data || []);
      })
      .catch((err) => console.error("CSV load error:", err))
      .finally(() => setLoading(false));
  }, []);

  const districtsStats = useMemo(() => {
    const map = {};
    rows.forEach((r) => {
      const district = (r.District || "").trim();
      if (!district) return;
      if (!map[district]) {
        map[district] = {
          totalCollisions: 0,
          trainCollisions: 0,
          roadCollisions: 0,
          vehicleTypes: new Set(),
          animalTypes: new Set(),
        };
      }
      const s = map[district];
      s.totalCollisions += 1;
      const vt = (r.vehicle_type || "").toLowerCase();
      if (vt === "train") s.trainCollisions += 1;
      else if (vt.includes("car") || vt.includes("bus") || vt.includes("vehicle")) s.roadCollisions += 1;
      s.vehicleTypes.add(r.vehicle_type || "Unknown");
      s.animalTypes.add(r.animal_type || "Unknown");
    });

    return Object.entries(map)
      .map(([district, s]) => ({
        district,
        totalCollisions: s.totalCollisions,
        trainCollisions: s.trainCollisions,
        roadCollisions: s.roadCollisions,
        vehicleTypeCount: s.vehicleTypes.size,
        animalTypeCount: s.animalTypes.size,
      }))
      .sort((a, b) => b.totalCollisions - a.totalCollisions);
  }, [rows]);

  const topDistricts = useMemo(() => districtsStats.slice(0, 10), [districtsStats]);

  if (loading) {
    return (
      <div className="dash">
        <Sidebar />
        <main className="dash__content">
          <div className="dash__loading">
            <div className="dash__loading-text">Loading district data…</div>
          </div>
        </main>
      </div>
    );
  }

  const avgCollisions = districtsStats.length > 0
    ? (rows.length / districtsStats.length).toFixed(1)
    : "—";

  return (
    <div className="dash">
      <Sidebar />

      <main className="dash__content">
        {/* ── Header ────────────────────────────────────────── */}
        <div className="dash__header">
          <div>
            <h2 className="dash__title">Top Dangerous Districts</h2>
            <p className="dash__subtitle">
              Districts ranked by total animal–vehicle collision count
            </p>
          </div>
          <div className="dash__badge">{districtsStats.length} districts analyzed</div>
        </div>

        {/* ── KPI Cards ──────────────────────────────────────── */}
        <section className="dash__kpis">
          <KpiCard
            icon="🏛️"
            title="Total Districts"
            value={districtsStats.length}
            description="In the dataset"
            badgeColor="#4ade80"
            badge="All"
          />
          <KpiCard
            icon="⚠️"
            title="Most Dangerous"
            value={topDistricts[0]?.district || "—"}
            description={topDistricts[0] ? `${topDistricts[0].totalCollisions} collisions` : "—"}
            badgeColor="#ef4444"
            badge="#1"
          />
          <KpiCard
            icon="📊"
            title="Total Collisions"
            value={rows.length.toLocaleString()}
            description="Across all districts"
            badgeColor="#60a5fa"
            badge="Total"
          />
          <KpiCard
            icon="📉"
            title="Avg per District"
            value={avgCollisions}
            description="Average collision count"
            badgeColor="#facc15"
            badge="Avg"
          />
        </section>

        {/* ── Panels ─────────────────────────────────────────── */}
        <section className="dash__grid">

          {/* Bar Chart – top 10 */}
          <div className="panel panel--span2">
            <div className="panel__head">
              <h3 className="panel__title">Top 10 Most Dangerous Districts</h3>
              <span className="panel__meta">Click bar to drill down</span>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart
                data={topDistricts}
                margin={{ top: 6, right: 16, left: -10, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="district"
                  tick={{ fill: "#e6edf3", fontSize: 12, fontWeight: 500 }}
                  axisLine={false}
                  tickLine={false}
                  interval={0}
                  angle={-30}
                  textAnchor="end"
                  height={56}
                />
                <YAxis
                  tick={{ fill: "#8b949e", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={38}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  wrapperStyle={{ fontSize: 11, color: "#8b949e", paddingTop: 10 }}
                  formatter={() => "Total collisions"}
                />
                <Bar
                  dataKey="totalCollisions"
                  name="Collisions"
                  radius={[6, 6, 0, 0]}
                  onClick={(d) => navigate(`/district/${encodeURIComponent(d.district)}`)}
                  style={{ cursor: "pointer" }}
                >
                  {topDistricts.map((_, i) => (
                    <Cell key={i} fill={DANGER_PALETTE[i % DANGER_PALETTE.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="panel__note">
              Bars colour-coded from most (red) to least (yellow) dangerous. Click any bar for detailed district analysis.
            </p>
          </div>

          {/* Ranking Table */}
          <div className="panel panel--span2">
            <div className="panel__head">
              <h3 className="panel__title">Detailed Ranking — Top 20 Districts</h3>
              <span className="panel__meta">Comprehensive breakdown</span>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table className="districts-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>District</th>
                    <th className="num">Total</th>
                    <th className="num">Train</th>
                    <th className="num">Road</th>
                    <th className="num">Vehicle Types</th>
                    <th className="num">Animal Types</th>
                    <th className="num">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {districtsStats.slice(0, 20).map((d, idx) => {
                    const severity = idx === 0
                      ? "#ef4444"
                      : idx < 3
                      ? "#f97316"
                      : idx < 6
                      ? "#facc15"
                      : "#4ade80";
                    return (
                      <tr
                        key={d.district}
                        className="districts-table__row"
                        onClick={() => navigate(`/district/${encodeURIComponent(d.district)}`)}
                      >
                        <td>
                          <span className="rank-badge" style={{ color: severity }}>
                            #{idx + 1}
                          </span>
                        </td>
                        <td>
                          <span className="district-name">{d.district}</span>
                        </td>
                        <td className="num" style={{ color: severity, fontWeight: 700 }}>
                          {d.totalCollisions}
                        </td>
                        <td className="num">{d.trainCollisions}</td>
                        <td className="num">{d.roadCollisions}</td>
                        <td className="num">{d.vehicleTypeCount}</td>
                        <td className="num">{d.animalTypeCount}</td>
                        <td className="num">
                          <button
                            className="view-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              navigate(`/district/${encodeURIComponent(d.district)}`);
                            }}
                          >
                            View →
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="panel__note">
              Click any row or "View →" to open the detailed district analysis page.
            </p>
          </div>

          {/* Summary stats */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Top 3 Hotspots</h3>
              <span className="panel__meta">Highest risk</span>
            </div>
            <div className="stats-grid">
              {topDistricts.slice(0, 3).map((d, i) => (
                <StatItem
                  key={d.district}
                  label={`#${i + 1} ${d.district}`}
                  value={d.totalCollisions}
                />
              ))}
              <StatItem label="Avg per District" value={avgCollisions} />
            </div>
          </div>

          {/* Collision distribution */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Collision Summary</h3>
              <span className="panel__meta">Overall stats</span>
            </div>
            <div className="stats-grid">
              <StatItem label="Total Incidents" value={rows.length.toLocaleString()} />
              <StatItem
                label="Train Collisions"
                value={rows.filter((r) => (r.vehicle_type || "").toLowerCase() === "train").length}
              />
              <StatItem label="Districts" value={districtsStats.length} />
              <StatItem label="Top District" value={topDistricts[0]?.district || "—"} />
            </div>
          </div>

        </section>
      </main>

      <style>{`
        .districts-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
          min-width: 600px;
        }
        .districts-table thead tr {
          border-bottom: 1px solid rgba(34, 197, 94, 0.2);
        }
        .districts-table th {
          padding: 10px 14px;
          text-align: left;
          font-size: 10px;
          font-weight: 700;
          color: #6e7681;
          text-transform: uppercase;
          letter-spacing: 0.8px;
          background: rgba(255,255,255,0.02);
          white-space: nowrap;
        }
        .districts-table th.num,
        .districts-table td.num {
          text-align: right;
        }
        .districts-table td {
          padding: 11px 14px;
          border-bottom: 1px solid rgba(255,255,255,0.04);
          color: #c9d1d9;
          vertical-align: middle;
        }
        .districts-table__row {
          cursor: pointer;
          transition: background 0.15s;
        }
        .districts-table__row:hover td {
          background: rgba(34, 197, 94, 0.06);
        }
        .rank-badge {
          font-weight: 800;
          font-size: 13px;
        }
        .district-name {
          font-weight: 600;
          color: #e6edf3;
        }
        .view-btn {
          background: rgba(34, 197, 94, 0.1);
          color: #4ade80;
          border: 1px solid rgba(34, 197, 94, 0.25);
          padding: 4px 10px;
          border-radius: 8px;
          cursor: pointer;
          font-size: 11px;
          font-weight: 600;
          transition: background 0.2s;
          white-space: nowrap;
        }
        .view-btn:hover {
          background: rgba(34, 197, 94, 0.2);
        }
      `}</style>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
function KpiCard({ icon, title, value, description, badge, badgeColor }) {
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
