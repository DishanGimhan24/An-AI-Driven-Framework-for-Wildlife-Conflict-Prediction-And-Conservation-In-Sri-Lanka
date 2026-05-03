// src/components/PredictForm.jsx
import { useState, useEffect } from "react";
import { api } from "../services/api";

const MONTHS = [
  "January","February","March","April","May","June",
  "July","August","September","October","November","December"
];

export default function PredictForm({ onSubmit, loading }) {
  const [regions,  setRegions]  = useState([]);
  const [form,     setForm]     = useState({
    region: "", month: "", recentOffences: 0, rainfallMm: "",
  });

  useEffect(() => {
    api.regions().then((d) => setRegions(d.regions || []));
  }, []);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  function handleSubmit(e) {
    e.preventDefault();
    if (!form.region || !form.month) return;
    onSubmit({
      region:         form.region,
      month:          parseInt(form.month),
      recentOffences: parseInt(form.recentOffences) || 0,
      rainfallMm:     form.rainfallMm !== "" ? parseFloat(form.rainfallMm) : null,
    });
  }

  return (
    <form className="predict-form" onSubmit={handleSubmit}>
      <div className="form-grid">
        {/* Region */}
        <div className="form-group">
          <label>📍 Region *</label>
          <select value={form.region} onChange={set("region")} required>
            <option value="">Select region…</option>
            {regions.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name} (rank #{r.hotspot_rank})
              </option>
            ))}
          </select>
        </div>

        {/* Month */}
        <div className="form-group">
          <label>📅 Month *</label>
          <select value={form.month} onChange={set("month")} required>
            <option value="">Select month…</option>
            {MONTHS.map((m, i) => (
              <option key={m} value={i + 1}>{m}</option>
            ))}
          </select>
        </div>

        {/* Recent offences */}
        <div className="form-group">
          <label>⚠️ Recent Offences (30 days)</label>
          <input
            type="number" min="0" max="200"
            value={form.recentOffences}
            onChange={set("recentOffences")}
            placeholder="0"
          />
          <small>Reported cases in this region recently.</small>
        </div>

        {/* Rainfall */}
        <div className="form-group">
          <label>🌧️ Rainfall (mm)</label>
          <input
            type="number" min="0" max="1000" step="0.1"
            value={form.rainfallMm}
            onChange={set("rainfallMm")}
            placeholder="Historical average"
          />
          <small>Optional. Actual rainfall in mm. Leave blank for default.</small>
        </div>
      </div>

      <button type="submit" className="btn-predict" disabled={loading}>
        {loading ? "🔄 Predicting…" : "🔮 Predict Risk"}
      </button>
    </form>
  );
}
