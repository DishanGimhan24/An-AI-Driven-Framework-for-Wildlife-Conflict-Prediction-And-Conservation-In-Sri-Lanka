const API_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8002";

async function http(path, options = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.detail?.message || data?.detail || data?.message || JSON.stringify(data);
    throw new Error(msg);
  }
  return data;
}

export const api = {
  getRegions: () => http("/regions"),
  getLocations: (region) => http(`/locations?region=${encodeURIComponent(region)}`),

  predict: (payload) => http("/predict", { method: "POST", body: JSON.stringify(payload) }),

  // community reports (we will add backend endpoints in section 3)
  createReport: (payload) => http("/reports", { method: "POST", body: JSON.stringify(payload) }),
  getMyReport: (reportId) => http(`/reports/${encodeURIComponent(reportId)}`),

  // officer
  officerLogin: (payload) => http("/auth/officer/login", { method: "POST", body: JSON.stringify(payload) }),
  listReports: (token) => http("/officer/reports", { headers: { Authorization: `Bearer ${token}` } }),
  updateReport: (token, id, payload) =>
    http(`/officer/reports/${encodeURIComponent(id)}`, {
      method: "PUT",
      headers: { Authorization: `Bearer ${token}` },
      body: JSON.stringify(payload),
    }),
};
