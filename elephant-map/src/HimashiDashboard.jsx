import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import HimashiSidebar from "./HimashiSidebar";
import "./HimashiDashboard.css";

export default function HimashiDashboard() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/risk_map_data.csv")
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

  const safeNum = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  };

  const summary = useMemo(() => {
    const total = rows.length;

    const highCount = rows.filter((r) => r.risk_level === "HIGH").length;
    const medCount  = rows.filter((r) => r.risk_level === "MEDIUM").length;
    const lowCount  = rows.filter((r) => r.risk_level === "LOW").length;

    const vehicleCounts = rows.reduce(
      (acc, r) => {
        const vt = (r.vehicle_type || "").toLowerCase();
        if (vt.includes("train") || vt.includes("rail")) acc.train += 1;
        else if (
          vt.includes("road") ||
          vt.includes("car") ||
          vt.includes("bus") ||
          vt.includes("van") ||
          vt.includes("bike")
        )
          acc.road += 1;
        else acc.other += 1;
        return acc;
      },
      { train: 0, road: 0, other: 0 }
    );

    const districtHigh = rows.reduce((acc, r) => {
      const d = (r.district || r.District || "").trim();
      if (!d || r.risk_level !== "HIGH") return acc;
      acc[d] = (acc[d] || 0) + 1;
      return acc;
    }, {});

    const topDistricts = Object.entries(districtHigh)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([district, count]) => ({ district, count }));

    const clusterStats = rows.reduce(
      (acc, r) => {
        const cid = r.cluster_id;
        if (!cid || cid === "-1") return acc;
        acc.unique.add(cid);
        acc.points += 1;
        acc.maxRisk = Math.max(acc.maxRisk, safeNum(r.risk_score));
        return acc;
      },
      { unique: new Set(), points: 0, maxRisk: 0 }
    );

    return {
      total,
      highCount,
      medCount,
      lowCount,
      vehicleCounts,
      topDistricts,
      clusterCount: clusterStats.unique.size,
      clusteredPoints: clusterStats.points,
      maxRisk: clusterStats.maxRisk,
    };
  }, [rows]);

  return (
    <div className="dash">
      <HimashiSidebar />

      <main className="dash__content">
        <div className="dash__header">
          <div>
            <h2 className="dash__title">Dashboard Overview</h2>
            <p className="dash__subtitle">
              Real-time summary from <b>risk_map_data.csv</b>
            </p>
          </div>
          <div className="dash__badge">
            {loading ? "Loading…" : `Loaded ${summary.total} records`}
          </div>
        </div>

        <section className="dash__kpis">
          <KpiCard title="High Risk Points" value={summary.highCount} hint="Risk level = HIGH" />
          <KpiCard title="Clusters Found" value={summary.clusterCount} hint="DBSCAN clusters (excluding -1)" />
          <KpiCard
            title="Max Risk Score"
            value={summary.maxRisk ? summary.maxRisk.toFixed(2) : "—"}
            hint="Highest predicted score"
          />
          <KpiCard
            title="Train vs Road"
            value={`${summary.vehicleCounts.train} / ${summary.vehicleCounts.road}`}
            hint="train / road"
          />
        </section>

        <section className="dash__grid">
          {/* Risk Level Breakdown */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Risk Level Breakdown</h3>
              <span className="panel__meta">Counts</span>
            </div>
            <div className="breakdown">
              <div className="pill pill--high">
                <span>HIGH</span>
                <b>{summary.highCount}</b>
              </div>
              <div className="pill pill--med">
                <span>MEDIUM</span>
                <b>{summary.medCount}</b>
              </div>
              <div className="pill pill--low">
                <span>LOW</span>
                <b>{summary.lowCount}</b>
              </div>
            </div>
            <p className="panel__note">
              These numbers are computed directly from your ML output file.
            </p>
          </div>

          {/* Dangerous Districts */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Top Dangerous Districts</h3>
              <span className="panel__meta">HIGH risk only</span>
            </div>
            {summary.topDistricts.length === 0 ? (
              <div className="empty">
                No district column found (or no HIGH records).
                <div className="empty__hint">
                  If you have a district field in CSV, name it <b>district</b>.
                </div>
              </div>
            ) : (
              <table className="table">
                <thead>
                  <tr>
                    <th>District</th>
                    <th style={{ textAlign: "right" }}>High-risk count</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.topDistricts.map((d) => (
                    <tr key={d.district}>
                      <td>{d.district}</td>
                      <td style={{ textAlign: "right" }}>{d.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <p className="panel__note">
              Use this for "dangerous district" page / alerts.
            </p>
          </div>

          {/* Vehicle Breakdown */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Vehicle Type Breakdown</h3>
              <span className="panel__meta">Train / Road / Other</span>
            </div>
            <div className="bars">
              <BarRow label="Train" value={summary.vehicleCounts.train} total={summary.total} />
              <BarRow label="Road"  value={summary.vehicleCounts.road}  total={summary.total} />
              <BarRow label="Other" value={summary.vehicleCounts.other} total={summary.total} />
            </div>
            <p className="panel__note">
              If your dataset has mixed incidents, this helps show distribution.
            </p>
          </div>

          {/* Cluster Summary */}
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Cluster Summary</h3>
              <span className="panel__meta">DBSCAN</span>
            </div>
            <div className="clusterBox">
              <div className="clusterBox__item">
                <span>Total clusters</span>
                <b>{summary.clusterCount}</b>
              </div>
              <div className="clusterBox__item">
                <span>Clustered points</span>
                <b>{summary.clusteredPoints}</b>
              </div>
              <div className="clusterBox__item">
                <span>Noise points</span>
                <b>{summary.total - summary.clusteredPoints}</b>
              </div>
            </div>
            <p className="panel__note">
              Clustered points are those with cluster_id ≠ -1.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

function KpiCard({ title, value, hint }) {
  return (
    <div className="kpi">
      <div className="kpi__title">{title}</div>
      <div className="kpi__value">{value}</div>
      <div className="kpi__hint">{hint}</div>
    </div>
  );
}

function BarRow({ label, value, total }) {
  const pct = total ? Math.round((value / total) * 100) : 0;
  return (
    <div className="barRow">
      <div className="barRow__top">
        <span className="barRow__label">{label}</span>
        <span className="barRow__val">
          {value} <span className="barRow__pct">({pct}%)</span>
        </span>
      </div>
      <div className="barRow__track">
        <div className="barRow__fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
