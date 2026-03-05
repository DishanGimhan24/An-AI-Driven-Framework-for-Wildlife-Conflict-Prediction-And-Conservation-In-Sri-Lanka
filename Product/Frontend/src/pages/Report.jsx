import { useEffect, useState } from "react";
import { api } from "../api/client";

export default function Report() {
  const [regions, setRegions] = useState([]);
  const [locations, setLocations] = useState([]);

  const [region, setRegion] = useState("");
  const [location, setLocation] = useState("");

  const [offenceType, setOffenceType] = useState("suspected_poaching");
  const [when, setWhen] = useState("");
  const [description, setDescription] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const data = await api.getRegions();
        setRegions(data.regions || []);
        if (data.regions?.length) setRegion(data.regions[0]);
      } catch (e) {
        setError(e.message);
      }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      if (!region) return;
      try {
        const data = await api.getLocations(region);
        setLocations(data.locations || []);
        if (data.locations?.length) setLocation(data.locations[0]);
      } catch (e) {
        setError(e.message);
      }
    })();
  }, [region]);

  async function submit() {
    setError("");
    setResult(null);

    if (!region || !location || !description.trim()) {
      setError("Please fill region, location, and description.");
      return;
    }

    setSubmitting(true);
    try {
      const payload = {
        region,
        location,
        offence_type: offenceType,
        occurred_at: when || null,
        description: description.trim(),
      };
      const data = await api.createReport(payload);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="report-container">
      <div className="report-header">
        <h1>📝 Community Report</h1>
        <p>Report illegal poaching or wildlife offences. Your report goes to wildlife officers.</p>
      </div>

      {error && <div className="error">{error}</div>}
      {result && (
        <div className="alert-success">
          <div className="alert-success-title">✓ Report submitted successfully!</div>
          <p>Report ID: <strong>{result.report_id}</strong></p>
          <p>Status: <strong>{result.status}</strong></p>
        </div>
      )}

      <div className="glass-card">
        <div className="report-form">
          <div className="form-group">
            <label>🌍 Region</label>
            <select value={region} onChange={(e) => setRegion(e.target.value)} className="glass-input">
              {regions.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>

          <div className="form-group">
            <label>📍 Location</label>
            <select value={location} onChange={(e) => setLocation(e.target.value)} className="glass-input">
              {locations.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>

          <div className="form-group">
            <label>⚠️ Offence type (what you suspect)</label>
            <select value={offenceType} onChange={(e) => setOffenceType(e.target.value)} className="glass-input">
              <option value="suspected_poaching">Suspected poaching</option>
              <option value="illegal_logging">Illegal logging</option>
              <option value="traps_or_weapons">Traps or weapons</option>
              <option value="meat_trade">Meat or egg trade</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="form-group">
            <label>📅 Date & time (optional)</label>
            <input value={when} onChange={(e) => setWhen(e.target.value)} placeholder="2026-01-05 13:30" className="glass-input" />
          </div>

          <div className="form-group">
            <label>📄 Description</label>
            <textarea
              rows={5}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What happened? Any vehicle number, people count, direction, sounds, or evidence?"
              className="glass-input"
            />
          </div>

          <button className="submit-btn" onClick={submit} disabled={submitting}>
            {submitting ? "Submitting..." : "Submit Report"}
          </button>
        </div>
      </div>
    </div>
  );
}
