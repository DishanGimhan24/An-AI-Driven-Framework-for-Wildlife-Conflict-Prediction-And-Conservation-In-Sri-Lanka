import { useEffect, useState } from "react";
import { api } from "../api/client";
import { RefreshCw } from "lucide-react";

export default function OfficerDashboard() {
  const [reports, setReports] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const token = localStorage.getItem("officer_token") || "";

  async function load() {
    setError("");
    setLoading(true);
    try {
      const data = await api.listReports(token);
      setReports(data.reports || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function updateStatus(id, status) {
    setError("");
    try {
      await api.updateReport(token, id, { status });
      await load();
    } catch (e) {
      setError(e.message);
    }
  }

  const getStatusClass = (status) => {
    const map = {
      pending: "status-pending",
      under_review: "status-review",
      dispatched: "status-dispatched",
      closed: "status-closed"
    };
    return map[status] || "status-pending";
  };

  return (
    <div className="officer-container">
      <div className="officer-header">
        <div>
          <h1>🛡️ Officer Dashboard</h1>
          <p>Incoming community reports - {reports.length} total</p>
        </div>
        <button className="refresh-btn" onClick={load} disabled={loading}>
          <RefreshCw size={18} />
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="reports-grid">
        {reports.length === 0 && !loading && (
          <div className="glass-card empty-state">
            <div className="empty-icon">📭</div>
            <h3>No Reports</h3>
            <p>There are no community reports at this time.</p>
          </div>
        )}

        {reports.map((r) => (
          <div className="report-card" key={r.report_id}>
            <div className="report-id">{r.report_id}</div>
            
            <div className="report-info">
              <div className="info-item">
                <span className="info-label">Region</span>
                <span className="info-value">{r.region}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Location</span>
                <span className="info-value">{r.location}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Type</span>
                <span className="info-value">{r.offence_type}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Status</span>
                <span className={`status-badge ${getStatusClass(r.status)}`}>{r.status}</span>
              </div>
            </div>

            <div className="report-actions">
              <button className="action-btn" onClick={() => updateStatus(r.report_id, "under_review")}>Review</button>
              <button className="action-btn" onClick={() => updateStatus(r.report_id, "dispatched")}>Dispatch</button>
              <button className="action-btn" onClick={() => updateStatus(r.report_id, "closed")}>Close</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
