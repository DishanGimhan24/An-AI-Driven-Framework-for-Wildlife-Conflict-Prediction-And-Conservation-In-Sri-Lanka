import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const MONTHS = [
  { v: 1, label: "1 - January" },
  { v: 2, label: "2 - February" },
  { v: 3, label: "3 - March" },
  { v: 4, label: "4 - April" },
  { v: 5, label: "5 - May" },
  { v: 6, label: "6 - June" },
  { v: 7, label: "7 - July" },
  { v: 8, label: "8 - August" },
  { v: 9, label: "9 - September" },
  { v: 10, label: "10 - October" },
  { v: 11, label: "11 - November" },
  { v: 12, label: "12 - December" },
];

function RiskBadge({ level }) {
  const cls =
    level === "High"
      ? "badge badge-high"
      : level === "Medium"
      ? "badge badge-medium"
      : "badge badge-low";
  return <span className={cls}>{level || "—"}</span>;
}

export default function App() {
  const [regions, setRegions] = useState([]);
  const [locations, setLocations] = useState([]);

  const [region, setRegion] = useState("");
  const [location, setLocation] = useState("");
  const [year, setYear] = useState(2026);
  const [month, setMonth] = useState(1);

  const [loadingPredict, setLoadingPredict] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [result, setResult] = useState(null);

  const [loadingHotspots, setLoadingHotspots] = useState(false);
  const [hotspotError, setHotspotError] = useState("");
  const [hotspots, setHotspots] = useState([]);

  const monthLabel = useMemo(() => {
    const m = MONTHS.find((x) => x.v === Number(month));
    return m ? m.label : "";
  }, [month]);

  // Load regions on start
  useEffect(() => {
    fetch(`${API_BASE}/regions`)
      .then((r) => r.json())
      .then((data) => {
        const list = data.regions || [];
        setRegions(list);
        if (list.length > 0) setRegion(list[0]);
      })
      .catch(() => {});
  }, []);

  // Load locations when region changes
  useEffect(() => {
    if (!region) return;
    fetch(`${API_BASE}/locations?region=${encodeURIComponent(region)}`)
      .then((r) => r.json())
      .then((data) => {
        const list = data.locations || [];
        setLocations(list);
        if (list.length > 0) setLocation(list[0]);
      })
      .catch(() => {});
  }, [region]);

  async function onPredict() {
    setPredictError("");
    setResult(null);
    setLoadingPredict(true);

    try {
      const payload = {
        region,
        location,
        year: Number(year),
        month: Number(month),
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setPredictError(String(e.message || e));
    } finally {
      setLoadingPredict(false);
    }
  }

  async function onLoadHotspots() {
    setHotspotError("");
    setHotspots([]);
    setLoadingHotspots(true);

    try {
      const url = `${API_BASE}/hotspots?year=${encodeURIComponent(
        Number(year)
      )}&month=${encodeURIComponent(Number(month))}&top_k=10`;

      const res = await fetch(url);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setHotspots(data.items || []);
    } catch (e) {
      setHotspotError(String(e.message || e));
    } finally {
      setLoadingHotspots(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>Wildlife Offence Early Warning</h1>
        <p>Select region and location, then predict risk and offence type. You can also generate hotspot rankings.</p>
      </header>

      <div className="grid">
        <section className="card">
          <h2>Prediction Input</h2>

          <div className="field">
            <label>Region</label>
            <select value={region} onChange={(e) => setRegion(e.target.value)}>
              {regions.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Location</label>
            <select value={location} onChange={(e) => setLocation(e.target.value)}>
              {locations.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </div>

          <div className="row">
            <div className="field">
              <label>Year</label>
              <input
                type="number"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                min={2000}
                max={2100}
              />
            </div>

            <div className="field">
              <label>Month</label>
              <select value={month} onChange={(e) => setMonth(e.target.value)}>
                {MONTHS.map((m) => (
                  <option key={m.v} value={m.v}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button className="btn" onClick={onPredict} disabled={loadingPredict}>
            {loadingPredict ? "Predicting..." : "Predict"}
          </button>

          {predictError ? <div className="error">{predictError}</div> : null}

          <div className="small-note">API: {API_BASE}</div>
        </section>

        <section className="card">
          <h2>Prediction Result</h2>

          {!result ? (
            <div className="muted">Submit a prediction to see results here.</div>
          ) : (
            <div className="result">
              <div className="kv">
                <div className="k">Location ID</div>
                <div className="v">{result.location_id}</div>
              </div>

              <div className="kv">
                <div className="k">Month</div>
                <div className="v">
                  {year} | {monthLabel}
                </div>
              </div>

              <div className="kv">
                <div className="k">Risk</div>
                <div className="v big">
                  {result.risk_percent}% <RiskBadge level={result.risk_level} />
                </div>
              </div>

              <div className="kv">
                <div className="k">Predicted offence type</div>
                <div className="v">{result.predicted_offence_type}</div>
              </div>

              <div className="kv">
                <div className="k">Top 3 offence types</div>
                <div className="v chips">
                  {(result.top3_offence_types || []).map((t) => (
                    <span key={t} className="chip">
                      {t}
                    </span>
                  ))}
                </div>
              </div>

              <div className="muted" style={{ marginTop: 10 }}>
                Tip: Use dropdowns to avoid spelling errors.
              </div>
            </div>
          )}
        </section>
      </div>

      <section className="card" style={{ marginTop: 18 }}>
        <div className="hotspot-head">
          <h2>Hotspot Ranking</h2>
          <button className="btn" onClick={onLoadHotspots} disabled={loadingHotspots}>
            {loadingHotspots ? "Loading..." : "Generate Top 10"}
          </button>
        </div>

        <div className="muted">
          Ranking for <b>{year}</b> and <b>{monthLabel}</b>.
        </div>

        {hotspotError ? <div className="error">{hotspotError}</div> : null}

        {hotspots.length === 0 ? (
          <div className="muted" style={{ marginTop: 10 }}>
            Click “Generate Top 10” to see the highest-risk locations for the selected month.
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Region</th>
                  <th>Location</th>
                  <th>Risk</th>
                  <th>Level</th>
                  <th>Likely offence</th>
                </tr>
              </thead>
              <tbody>
                {hotspots.map((h) => (
                  <tr key={`${h.region}-${h.location}-${h.rank}`}>
                    <td>{h.rank}</td>
                    <td>{h.region}</td>
                    <td>{h.location}</td>
                    <td>{h.risk_percent}%</td>
                    <td>
                      <RiskBadge level={h.risk_level} />
                    </td>
                    <td>{h.predicted_offence_type}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
