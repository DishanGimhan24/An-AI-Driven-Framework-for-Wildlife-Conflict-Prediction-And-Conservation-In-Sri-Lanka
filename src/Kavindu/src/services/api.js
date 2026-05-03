// src/services/api.js
// All API calls to the FastAPI backend live here.
// Change BASE_URL if you deploy the backend elsewhere.

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8003/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "API error");
  }
  return res.json();
}

export const api = {
  health:       ()         => request("/health"),
  regions:      ()         => request("/regions"),
  hotspots:     (n = 13)  => request(`/hotspots?top_n=${n}`),
  calendar:     ()         => request("/calendar"),
  offenceTypes: ()         => request("/offence-types"),
  batchPredict: (month)    => request(`/batch-predict/${month}`),
  regionProfile:(name)     => request(`/region/${encodeURIComponent(name)}/profile`),

  predict: ({ region, month, recentOffences = 0, rainfallMm = null }) =>
    request("/predict", {
      method: "POST",
      body: JSON.stringify({
        region,
        month,
        recent_offence_count: recentOffences,
        rainfall_mm: rainfallMm,
      }),
    }),
};
