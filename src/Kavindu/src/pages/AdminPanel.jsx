import { useEffect, useState } from "react";
import axios from "axios";
import { UserPlus, Edit2, Power, RefreshCw, Shield, Users, UserCheck, UserX, X, Search } from "lucide-react";

const API = "http://127.0.0.1:8000";

const ROLES = ["officer", "admin"];

function OfficerFormModal({ officer, regions, onClose, onSaved }) {
  const isEdit = !!officer;
  const [form, setForm] = useState({
    name: officer?.name || "",
    email: officer?.email || "",
    password: "",
    role: officer?.role || "officer",
    region: officer?.region || "",
    badge_number: officer?.badge_number || "",
    phone: officer?.phone || "",
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  function set(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setSaving(true);
    try {
      if (isEdit) {
        const updates = {};
        if (form.name) updates.name = form.name;
        if (form.email) updates.email = form.email;
        if (form.role) updates.role = form.role;
        if (form.region !== undefined) updates.region = form.region || null;
        if (form.badge_number !== undefined) updates.badge_number = form.badge_number || null;
        if (form.phone !== undefined) updates.phone = form.phone || null;
        await axios.patch(`${API}/api/admin/officers/${officer._id}`, updates);
      } else {
        if (!form.name || !form.email || !form.password) {
          setError("Name, email and password are required.");
          setSaving(false);
          return;
        }
        await axios.post(`${API}/api/admin/officers`, form);
      }
      onSaved();
      onClose();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" style={{ maxWidth: 520 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2 className="modal-title">{isEdit ? "Edit Officer" : "Add New Officer"}</h2>
            <p className="modal-subtitle">{isEdit ? `Editing ${officer.name}` : "Create a new officer account"}</p>
          </div>
          <button className="modal-close-btn" onClick={onClose}><X size={20} /></button>
        </div>

        <form onSubmit={handleSubmit} className="admin-form">
          {error && <div className="error" style={{ margin: "0 0 12px" }}>{error}</div>}

          <div className="admin-form-grid">
            <div className="form-group">
              <label>Full Name *</label>
              <input className="glass-input" value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="e.g. Kamal Perera" required={!isEdit} />
            </div>

            <div className="form-group">
              <label>Email *</label>
              <input className="glass-input" type="email" value={form.email} onChange={(e) => set("email", e.target.value)} placeholder="officer@dwd.lk" required={!isEdit} />
            </div>

            {!isEdit && (
              <div className="form-group">
                <label>Password *</label>
                <input className="glass-input" type="password" value={form.password} onChange={(e) => set("password", e.target.value)} placeholder="Set initial password" required />
              </div>
            )}

            <div className="form-group">
              <label>Role</label>
              <select className="glass-input" value={form.role} onChange={(e) => set("role", e.target.value)}>
                {ROLES.map((r) => <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>)}
              </select>
            </div>

            <div className="form-group">
              <label>Region</label>
              <select className="glass-input" value={form.region} onChange={(e) => set("region", e.target.value)}>
                <option value="">-- No region assigned --</option>
                {regions.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>

            <div className="form-group">
              <label>Badge Number</label>
              <input className="glass-input" value={form.badge_number} onChange={(e) => set("badge_number", e.target.value)} placeholder="e.g. WO-0042" />
            </div>

            <div className="form-group">
              <label>Phone</label>
              <input className="glass-input" value={form.phone} onChange={(e) => set("phone", e.target.value)} placeholder="e.g. +94 77 123 4567" />
            </div>
          </div>

          <div className="modal-actions" style={{ padding: 0, marginTop: 20 }}>
            <button type="button" className="modal-cancel-btn" onClick={onClose}>Cancel</button>
            <button type="submit" className="modal-assign-btn" disabled={saving}>
              {isEdit ? <Edit2 size={16} /> : <UserPlus size={16} />}
              {saving ? "Saving..." : isEdit ? "Save Changes" : "Create Officer"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default function AdminPanel() {
  const [officers, setOfficers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [regions, setRegions] = useState([]);
  const [showAdd, setShowAdd] = useState(false);
  const [editingOfficer, setEditingOfficer] = useState(null);
  const [search, setSearch] = useState("");
  const [togglingId, setTogglingId] = useState(null);

  async function load() {
    setLoading(true);
    setError("");
    try {
      const [oRes, rRes] = await Promise.all([
        axios.get(`${API}/api/admin/officers`),
        fetch(`${API}/regions`).then((r) => r.json()),
      ]);
      setOfficers(oRes.data.officers || []);
      setRegions(rRes.regions || []);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function toggleActive(officer) {
    setTogglingId(officer._id);
    try {
      await axios.patch(`${API}/api/admin/officers/${officer._id}/toggle`);
      await load();
    } catch (e) {
      alert(e.response?.data?.detail || e.message);
    } finally {
      setTogglingId(null);
    }
  }

  const filtered = officers.filter((o) => {
    const q = search.toLowerCase();
    return (
      o.name?.toLowerCase().includes(q) ||
      o.email?.toLowerCase().includes(q) ||
      o.badge_number?.toLowerCase().includes(q) ||
      o.region?.toLowerCase().includes(q)
    );
  });

  const totalActive = officers.filter((o) => o.is_active !== false).length;
  const totalInactive = officers.length - totalActive;

  function getInitials(name) {
    return (name || "?").split(" ").map((w) => w[0]).join("").toUpperCase().slice(0, 2);
  }

  return (
    <div className="admin-container">
      {(showAdd || editingOfficer) && (
        <OfficerFormModal
          officer={editingOfficer}
          regions={regions}
          onClose={() => { setShowAdd(false); setEditingOfficer(null); }}
          onSaved={load}
        />
      )}

      {/* Header */}
      <div className="admin-header">
        <div>
          <h1 className="admin-title">Admin Panel</h1>
          <p className="admin-subtitle">Manage wildlife officer accounts</p>
        </div>
        <div style={{ display: "flex", gap: "10px" }}>
          <button className="refresh-btn" onClick={load} disabled={loading}>
            <RefreshCw size={16} />
            {loading ? "Loading..." : "Refresh"}
          </button>
          <button className="admin-add-btn" onClick={() => setShowAdd(true)}>
            <UserPlus size={16} />
            Add Officer
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="admin-stats">
        <div className="admin-stat-card">
          <div className="admin-stat-icon" style={{ background: "rgba(16,185,129,0.15)", color: "var(--emerald-400)" }}>
            <Users size={22} />
          </div>
          <div>
            <div className="admin-stat-value">{officers.length}</div>
            <div className="admin-stat-label">Total Officers</div>
          </div>
        </div>
        <div className="admin-stat-card">
          <div className="admin-stat-icon" style={{ background: "rgba(99,102,241,0.15)", color: "#a5b4fc" }}>
            <Shield size={22} />
          </div>
          <div>
            <div className="admin-stat-value">{officers.filter((o) => o.role === "admin").length}</div>
            <div className="admin-stat-label">Admins</div>
          </div>
        </div>
        <div className="admin-stat-card">
          <div className="admin-stat-icon" style={{ background: "rgba(16,185,129,0.15)", color: "#34d399" }}>
            <UserCheck size={22} />
          </div>
          <div>
            <div className="admin-stat-value">{totalActive}</div>
            <div className="admin-stat-label">Active</div>
          </div>
        </div>
        <div className="admin-stat-card">
          <div className="admin-stat-icon" style={{ background: "rgba(239,68,68,0.15)", color: "#f87171" }}>
            <UserX size={22} />
          </div>
          <div>
            <div className="admin-stat-value">{totalInactive}</div>
            <div className="admin-stat-label">Inactive</div>
          </div>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {/* Search */}
      <div className="glass-card" style={{ padding: "16px 20px", marginBottom: 16 }}>
        <div className="admin-search-wrap">
          <Search size={16} className="admin-search-icon" />
          <input
            className="admin-search-input"
            placeholder="Search by name, email, badge or region..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Officer Table */}
      <div className="glass-card" style={{ padding: 0, overflow: "hidden" }}>
        <div className="admin-table-wrapper">
          <table className="admin-table">
            <thead>
              <tr>
                <th>Officer</th>
                <th>Badge</th>
                <th>Region</th>
                <th>Role</th>
                <th>Status</th>
                <th>Joined</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ textAlign: "center", padding: "40px", color: "#6b7280" }}>
                    {search ? "No officers match your search." : "No officers found."}
                  </td>
                </tr>
              )}
              {filtered.map((o) => {
                const isActive = o.is_active !== false;
                const isToggling = togglingId === o._id;
                return (
                  <tr key={o._id} className={`admin-table-row ${!isActive ? "admin-row-inactive" : ""}`}>
                    <td>
                      <div className="admin-officer-cell">
                        <div className="admin-avatar">{getInitials(o.name)}</div>
                        <div>
                          <div className="admin-officer-name">{o.name}</div>
                          <div className="admin-officer-email">{o.email}</div>
                        </div>
                      </div>
                    </td>
                    <td className="admin-td-muted">{o.badge_number || "—"}</td>
                    <td className="admin-td-muted">{o.region || "—"}</td>
                    <td>
                      <span className={`admin-role-badge ${o.role === "admin" ? "admin-role-admin" : "admin-role-officer"}`}>
                        {o.role === "admin" ? <Shield size={11} /> : <UserCheck size={11} />}
                        {o.role}
                      </span>
                    </td>
                    <td>
                      <span className={`admin-status-badge ${isActive ? "admin-status-active" : "admin-status-inactive"}`}>
                        {isActive ? "Active" : "Inactive"}
                      </span>
                    </td>
                    <td className="admin-td-muted">
                      {o.created_at ? new Date(o.created_at).toLocaleDateString() : "—"}
                    </td>
                    <td>
                      <div className="admin-actions-cell">
                        <button
                          className="admin-action-btn admin-edit-btn"
                          onClick={() => setEditingOfficer(o)}
                          title="Edit officer"
                        >
                          <Edit2 size={14} />
                        </button>
                        <button
                          className={`admin-action-btn ${isActive ? "admin-deactivate-btn" : "admin-activate-btn"}`}
                          onClick={() => toggleActive(o)}
                          disabled={isToggling}
                          title={isActive ? "Deactivate" : "Activate"}
                        >
                          <Power size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
