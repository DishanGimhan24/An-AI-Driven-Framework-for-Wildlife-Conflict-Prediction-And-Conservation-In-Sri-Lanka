import { useEffect, useState } from "react";
import axios from "axios";
import { RefreshCw, Users, X, Star, UserCheck, Wand2, Trash2 } from "lucide-react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import MarkerClusterGroup from "react-leaflet-cluster";
import "leaflet/dist/leaflet.css";
import "react-leaflet-cluster/dist/assets/MarkerCluster.css";
import "react-leaflet-cluster/dist/assets/MarkerCluster.Default.css";
import L from "leaflet";

// Status → color mapping
const STATUS_COLORS = {
  pending: { fill: "#ef4444", border: "#b91c1c" },
  under_review: { fill: "#f97316", border: "#c2410c" },
  investigating: { fill: "#3b82f6", border: "#1d4ed8" },
  resolved: { fill: "#22c55e", border: "#15803d" },
  closed: { fill: "#64748b", border: "#334155" },
};

function makeStatusIcon(status) {
  const { fill, border } = STATUS_COLORS[status] || STATUS_COLORS.pending;
  const isPending = status === "pending";
  const pulse = isPending
    ? `<span style="position:absolute;inset:-5px;border-radius:50%;border:2px solid ${fill};opacity:0.5;animation:map-pulse 1.5s ease-out infinite;"></span>`
    : "";
  return L.divIcon({
    className: "",
    html: `<span style="position:relative;display:flex;align-items:center;justify-content:center;">
      ${pulse}
      <span style="width:14px;height:14px;border-radius:50%;background:${fill};border:2px solid ${border};box-shadow:0 0 6px ${fill}88;display:block;"></span>
    </span>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [0, -14],
  });
}

// Coordinate mappings for Sri Lankan regions/national parks
const locationMap = {
  // Regions
  "Anuradhapura": [8.3114, 80.4037],
  "Central": [7.2906, 80.6337],
  "Eastern": [7.7170, 81.6989],
  "Head Office-Flying Squad": [6.8970, 79.9223],
  "Kilinochchi": [9.3803, 80.3770],
  "Polonnaruwa": [7.9403, 81.0188],
  "Puttalam": [8.0362, 79.8283],
  "Southern": [6.0535, 80.2210],
  "Trincomalee": [8.5811, 81.2330],
  "Uwa": [6.9847, 81.0549],
  "Vauniya": [8.7514, 80.4971],
  "Wayamba": [7.4818, 80.3609],
  "Western": [6.9271, 79.8612],

  // Specific Known Locations & Parks
  "Colombo": [6.9271, 79.8612],
  "Galle Face": [6.9238, 79.8458],
  "Kandy": [7.2906, 80.6337],
  "Yala NP": [6.3683, 81.5156],
  "Wilpattu NP": [8.4389, 79.9950],
  "Udawalawe NP": [6.4385, 80.8930],
  "Minneriya NP": [8.0319, 80.8351],
  "Horowpothana NP": [8.6946, 80.8652],
  "Bundala NP": [6.1950, 81.2330],
  "Gal Oya NP": [7.2140, 81.4111],
  "Lunugamwehera NP": [6.3869, 81.1966],
  "Wasgamuwa NP": [7.7126, 80.9329],
  "Angammadilla NP": [7.8288, 80.9576],
  "Kaudulla NP": [8.1818, 80.9258],
  "Somawathiya NP": [8.1396, 81.1970],
  "Hikkaduwa NP": [6.1438, 80.0963],
  "Pigeon Island": [8.7214, 81.1989]
};

function getCoords(region, location) {
  if (locationMap[location]) return locationMap[location];
  if (locationMap[region]) return locationMap[region];
  // Fallback to center of Sri Lanka with a tiny randomized jitter to prevent perfectly overlapping pins
  return [7.8731 + (Math.random() - 0.5) * 0.1, 80.7718 + (Math.random() - 0.5) * 0.1];
}

const ROLE_TABS = ["all", "officer", "admin"];

function AssignModal({ report, onClose, onAssigned }) {
  const [officers, setOfficers] = useState([]);
  const [selected, setSelected] = useState(report.assigned_team || []);
  const [teamLead, setTeamLead] = useState(report.team_lead || "");
  const [loadingOfficers, setLoadingOfficers] = useState(true);
  const [saving, setSaving] = useState(false);
  const [filterRegion, setFilterRegion] = useState(true);
  const [roleFilter, setRoleFilter] = useState("officer");
  const [emailSearch, setEmailSearch] = useState("");
  const [suggesting, setSuggesting] = useState(false);

  useEffect(() => {
    async function fetchOfficers() {
      setLoadingOfficers(true);
      try {
        const url = filterRegion
          ? `http://127.0.0.1:8000/api/officers?region=${encodeURIComponent(report.region)}`
          : "http://127.0.0.1:8000/api/officers";
        const res = await axios.get(url);
        setOfficers(res.data.officers || []);
      } catch {
        setOfficers([]);
      } finally {
        setLoadingOfficers(false);
      }
    }
    fetchOfficers();
  }, [filterRegion, report.region]);

  // Apply role + email search filters client-side
  const visibleOfficers = officers.filter((o) => {
    const matchRole = roleFilter === "all" || o.role === roleFilter;
    const q = emailSearch.trim().toLowerCase();
    const matchEmail = !q || o.email.toLowerCase().includes(q) || o.name.toLowerCase().includes(q);
    return matchRole && matchEmail;
  });

  function toggleOfficer(email) {
    setSelected((prev) => {
      const next = prev.includes(email) ? prev.filter((e) => e !== email) : [...prev, email];
      if (teamLead && !next.includes(teamLead)) setTeamLead("");
      return next;
    });
  }

  async function handleAssign() {
    setSaving(true);
    try {
      await axios.patch(`http://127.0.0.1:8000/api/reports/${report._id}/assign`, {
        assigned_team: selected,
        team_lead: teamLead || null,
      });
      onAssigned();
      onClose();
    } catch (e) {
      alert(e.response?.data?.detail || e.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleSuggest() {
    setSuggesting(true);
    try {
      const res = await axios.get(
        `http://127.0.0.1:8000/api/officers/suggest?region=${encodeURIComponent(report.region)}&offence_type=${encodeURIComponent(report.offence_type)}`
      );
      const { suggestions, suggested_lead } = res.data;
      const emails = suggestions.map((o) => o.email);
      setSelected(emails);
      setTeamLead(suggested_lead || "");
      // Ensure all suggested officers are visible (turn off region filter if needed)
      if (suggestions.some((o) => o.region !== report.region)) {
        setFilterRegion(false);
      }
    } catch (e) {
      alert(e.response?.data?.detail || e.message);
    } finally {
      setSuggesting(false);
    }
  }

  function handleClear() {
    setSelected([]);
    setTeamLead("");
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2 className="modal-title">Assign Investigation Team</h2>
            <p className="modal-subtitle">{report.reference_id} &bull; {report.location} ({report.region})</p>
          </div>
          <button className="modal-close-btn" onClick={onClose}><X size={20} /></button>
        </div>

        {/* Role filter tabs + region toggle */}
        <div className="modal-filter-row" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
            <div className="assign-role-tabs">
              {ROLE_TABS.map((r) => (
                <button
                  key={r}
                  className={`assign-role-tab ${roleFilter === r ? "assign-role-tab-active" : ""}`}
                  onClick={() => setRoleFilter(r)}
                >
                  {r.charAt(0).toUpperCase() + r.slice(1)}
                  <span className="assign-role-count">
                    {r === "all" ? officers.length : officers.filter((o) => o.role === r).length}
                  </span>
                </button>
              ))}
            </div>
            <button
              className="suggest-btn"
              onClick={handleSuggest}
              disabled={suggesting || loadingOfficers}
              title={`Auto-suggest team for ${report.offence_type.replace(/_/g, " ")} in ${report.region}`}
            >
              <Wand2 size={14} />
              {suggesting ? "Suggesting..." : "Auto-suggest"}
            </button>
          </div>

          {/* Email autocomplete search */}
          <div style={{ position: "relative" }}>
            <input
              className="assign-email-search"
              list="officer-email-list"
              placeholder="Search by name or email..."
              value={emailSearch}
              onChange={(e) => setEmailSearch(e.target.value)}
              autoComplete="off"
            />
            <datalist id="officer-email-list">
              {officers
                .filter((o) => roleFilter === "all" || o.role === roleFilter)
                .map((o) => (
                  <option key={o._id} value={o.email}>{o.name} — {o.email}</option>
                ))}
            </datalist>
          </div>

          <label className="modal-filter-label" style={{ marginTop: 2 }}>
            <input
              type="checkbox"
              checked={filterRegion}
              onChange={(e) => setFilterRegion(e.target.checked)}
              style={{ marginRight: 8 }}
            />
            Show only officers from <strong style={{ color: "var(--emerald-400)", marginLeft: 4 }}>{report.region}</strong>
          </label>
        </div>

        <div className="modal-officers-list">
          {loadingOfficers ? (
            <p className="modal-loading">Loading officers...</p>
          ) : visibleOfficers.length === 0 ? (
            <p className="modal-loading">No officers match the current filters.</p>
          ) : (
            visibleOfficers.map((o) => {
              const isSelected = selected.includes(o.email);
              const isLead = teamLead === o.email;
              return (
                <div key={o._id} className={`officer-row ${isSelected ? "officer-row-selected" : ""}`}>
                  <label className="officer-check-label">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleOfficer(o.email)}
                    />
                    <div className="officer-info">
                      <span className="officer-name">
                        {o.name}
                        <span className={`assign-role-pill ${o.role === "admin" ? "assign-role-pill-admin" : "assign-role-pill-officer"}`}>
                          {o.role}
                        </span>
                        {isLead && <span className="lead-star"><Star size={12} /> Lead</span>}
                      </span>
                      <span className="officer-meta">{o.email}{o.badge_number ? ` • Badge: ${o.badge_number}` : ""}{o.region ? ` • ${o.region}` : ""}</span>
                    </div>
                  </label>
                  {isSelected && (
                    <button
                      className={`lead-btn ${isLead ? "lead-btn-active" : ""}`}
                      onClick={() => setTeamLead(isLead ? "" : o.email)}
                      title="Set as team lead"
                    >
                      <Star size={14} />
                    </button>
                  )}
                </div>
              );
            })
          )}
        </div>

        {selected.length > 0 && (
          <div className="modal-selected-summary">
            <UserCheck size={16} />
            <span>{selected.length} officer{selected.length > 1 ? "s" : ""} selected</span>
            {teamLead && <span className="summary-lead">&bull; Lead: {officers.find((o) => o.email === teamLead)?.name || teamLead}</span>}
            <button className="clear-team-btn" onClick={handleClear} title="Clear all selections">
              <Trash2 size={13} />
              Clear
            </button>
          </div>
        )}

        <div className="modal-actions">
          <button className="modal-cancel-btn" onClick={onClose}>Cancel</button>
          <button className="modal-assign-btn" onClick={handleAssign} disabled={saving || selected.length === 0}>
            <Users size={16} />
            {saving ? "Assigning..." : "Assign Team"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function OfficerDashboard() {
  const [reports, setReports] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [assigningReport, setAssigningReport] = useState(null);

  async function load() {
    setError("");
    setLoading(true);
    try {
      const response = await axios.get("http://127.0.0.1:8000/api/reports");
      setReports(response.data.reports || []);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function updateStatus(id, status) {
    if (!id) return;
    setError("");
    try {
      await axios.patch(`http://127.0.0.1:8000/api/reports/${id}/status`, { status });
      await load();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
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
      {assigningReport && (
        <AssignModal
          report={assigningReport}
          onClose={() => setAssigningReport(null)}
          onAssigned={load}
        />
      )}

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

      {/* Map legend */}
      <div className="map-legend">
        {Object.entries(STATUS_COLORS).map(([status, { fill }]) => (
          <span key={status} className="map-legend-item">
            <span className="map-legend-dot" style={{ background: fill, boxShadow: `0 0 5px ${fill}88` }} />
            {status.replace("_", " ")}
          </span>
        ))}
      </div>

      <div className="glass-card" style={{ height: "480px", marginBottom: "2rem", overflow: "hidden", padding: 0 }}>
        <MapContainer center={[7.8731, 80.7718]} zoom={7} style={{ height: "100%", width: "100%" }}>
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
          />
          <MarkerClusterGroup chunkedLoading>
            {reports.map((r) => {
              const [lat, lng] = (r.latitude && r.longitude) ? [r.latitude, r.longitude] : getCoords(r.region, r.location);
              return (
                <Marker position={[lat, lng]} key={`map-${r._id}`} icon={makeStatusIcon(r.status)}>
                  <Popup>
                    <div style={{ minWidth: 200, fontFamily: "inherit" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                        <strong style={{ fontSize: 13 }}>{r.reference_id || r._id}</strong>
                        <span style={{
                          background: STATUS_COLORS[r.status]?.fill || "#888",
                          color: "#fff",
                          fontSize: 10,
                          padding: "2px 7px",
                          borderRadius: 10,
                          textTransform: "capitalize",
                          fontWeight: 600,
                        }}>
                          {r.status.replace(/_/g, " ")}
                        </span>
                      </div>
                      <div style={{ fontSize: 12, color: "#555", lineHeight: 1.7 }}>
                        <div><b>Type:</b> <span style={{ textTransform: "capitalize" }}>{r.offence_type.replace(/_/g, " ")}</span></div>
                        <div><b>Location:</b> {r.location} &bull; {r.region}</div>
                        <div>
                          <b>Team:</b>{" "}
                          {r.assigned_team?.length
                            ? <span style={{ color: "#16a34a" }}>{r.assigned_team.length} officer{r.assigned_team.length > 1 ? "s" : ""} assigned</span>
                            : <span style={{ color: "#dc2626" }}>Unassigned</span>}
                        </div>
                        {r.team_lead && (
                          <div>
                            <b>Lead:</b> {r.team_lead_name || r.team_lead}
                            {r.team_lead_phone && <span style={{ color: "var(--emerald-400)", marginLeft: "4px" }}>📞 {r.team_lead_phone}</span>}
                          </div>
                        )}
                      </div>
                      <button
                        onClick={() => setAssigningReport(r)}
                        style={{
                          marginTop: 10, width: "100%", padding: "5px 0",
                          background: r.assigned_team?.length ? "rgba(99,102,241,0.12)" : "rgba(239,68,68,0.1)",
                          border: `1px solid ${r.assigned_team?.length ? "rgba(99,102,241,0.4)" : "rgba(239,68,68,0.35)"}`,
                          borderRadius: 6, cursor: "pointer", fontSize: 12, fontWeight: 600,
                          color: r.assigned_team?.length ? "#818cf8" : "#f87171",
                        }}
                      >
                        {r.assigned_team?.length ? "Edit Team" : "Assign Team"}
                      </button>
                    </div>
                  </Popup>
                </Marker>
              );
            })}
          </MarkerClusterGroup>
        </MapContainer>
      </div>

      <div className="reports-grid">
        {reports.length === 0 && !loading && (
          <div className="glass-card empty-state">
            <div className="empty-icon">📭</div>
            <h3>No Reports</h3>
            <p>There are no community reports at this time.</p>
          </div>
        )}

        {reports.map((r) => (
          <div className="glass-card" style={{ padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }} key={r._id}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontSize: "1.2rem", fontWeight: "bold" }}>{r.reference_id || r._id}</div>
              <span className={`badge ${r.status !== 'pending' ? 'badge-high' : 'badge-medium'}`} style={{ textTransform: "capitalize" }}>
                {r.status.replace("_", " ")}
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: "0.9rem", color: "#ddd" }}>
              <div><strong>Region:</strong> {r.region}</div>
              <div><strong>Location:</strong> {r.location}</div>
              <div><strong>Type:</strong> {r.offence_type.replace("_", " ")}</div>
              <div><strong>Date:</strong> {r.incident_datetime || "Not provided"}</div>
              <div style={{ gridColumn: "span 2" }}><strong>Created:</strong> {new Date(r.created_at).toLocaleString()}</div>
            </div>

            <div style={{ background: "rgba(0,0,0,0.2)", padding: "1rem", borderRadius: "8px", fontSize: "0.95rem" }}>
              <strong>Description:</strong><br />
              {r.description}
            </div>

            {r.image_url && (
              <div style={{ marginTop: "0.5rem" }}>
                <strong>Evidence Photo:</strong>
                <img
                  src={`http://127.0.0.1:8000${r.image_url}`}
                  alt="Evidence"
                  style={{ width: "100%", maxHeight: "250px", objectFit: "cover", borderRadius: "8px", marginTop: "0.5rem", border: "1px solid rgba(255,255,255,0.1)" }}
                />
              </div>
            )}

            {r.assigned_team && r.assigned_team.length > 0 && (
              <div className="assigned-team-display">
                <div className="assigned-team-label">
                  <Users size={14} /> Assigned Team
                </div>
                <div className="assigned-team-members">
                  {r.assigned_team.map((email) => (
                    <span key={email} className={`team-member-chip ${r.team_lead === email ? "team-lead-chip" : ""}`}>
                      {r.team_lead === email && <Star size={11} />}
                      {email}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: "flex", gap: "0.5rem", marginTop: "auto", flexWrap: "wrap" }}>
              <button className="submit-btn" style={{ padding: "0.5rem" }} onClick={() => updateStatus(r._id, "under_review")} disabled={r.status === "under_review"}>Review</button>
              <button className="submit-btn" style={{ padding: "0.5rem" }} onClick={() => updateStatus(r._id, "investigating")} disabled={r.status === "investigating"}>Investigate</button>
              <button className="submit-btn" style={{ padding: "0.5rem", background: "rgba(255,100,100,0.2)" }} onClick={() => updateStatus(r._id, "resolved")} disabled={r.status === "resolved"}>Resolve</button>
              <button
                className="submit-btn"
                style={{ padding: "0.5rem", background: "rgba(99,102,241,0.2)", border: "1px solid rgba(99,102,241,0.4)", display: "flex", alignItems: "center", gap: "6px" }}
                onClick={() => setAssigningReport(r)}
              >
                <Users size={15} />
                {r.assigned_team?.length ? "Edit Team" : "Assign Team"}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
