// src/components/RiskCard.jsx
// Shows the main prediction result — warning level, risk %, offence type.

export default function RiskCard({ result }) {
  if (!result) return null;

  const colors = {
    CRITICAL: { bg: "rgba(239, 68, 68, 0.04)", border: "rgba(239, 68, 68, 0.3)", text: "#f87171", fill: "#ef4444", glow: "0 0 20px rgba(239, 68, 68, 0.15)" },
    HIGH:     { bg: "rgba(249, 115, 22, 0.04)", border: "rgba(249, 115, 22, 0.3)", text: "#fb923c", fill: "#f97316", glow: "0 0 20px rgba(249, 115, 22, 0.15)" },
    MEDIUM:   { bg: "rgba(245, 158, 11, 0.04)", border: "rgba(245, 158, 11, 0.3)", text: "#fbbf24", fill: "#f59e0b", glow: "0 0 20px rgba(245, 158, 11, 0.15)" },
    LOW:      { bg: "rgba(34, 197, 94, 0.04)", border: "rgba(34, 197, 94, 0.3)", text: "#4ade80", fill: "#22c55e", glow: "0 0 20px rgba(34, 197, 94, 0.15)" },
  };
  const c = colors[result.warning] || colors.LOW;

  return (
    <div className="risk-card" style={{ borderColor: c.border, background: c.bg, boxShadow: c.glow }}>
      {/* Header */}
      <div className="risk-header">
        <div className="risk-icon-wrapper" style={{ background: c.fill, boxShadow: c.glow }}>
          <span className="risk-emoji">{result.warning_emoji}</span>
        </div>
        <div className="risk-title-group">
          <div className="risk-label" style={{ color: c.text }}>
            {result.warning} RISK
          </div>
          <div className="risk-location">
            {result.region} · {result.month_name} · {result.season} Season
          </div>
        </div>
        <div className="risk-pct-circle" style={{ borderColor: c.border, color: c.text, background: c.bg, boxShadow: c.glow }}>
          <span>{result.risk_pct}%</span>
          <small>risk</small>
        </div>
      </div>

      {/* Key facts */}
      <div className="risk-facts">
        <Fact label="Hotspot Rank" value={`#${result.hotspot_rank}`} />
        <Fact label="Rainfall"     value={`${result.rainfall_mm} mm${result.is_drought ? " ⚠️ drought" : ""}`} />
        <Fact label="Top Offence"  value={result.single_offence} />
        <Fact label="Also Likely"  value={result.multi_offences.slice(0,2).join(", ")} />
      </div>

      {/* Probability bars */}
      <div className="prob-section">
        <h4>Offence Group Probabilities</h4>
        <div className="prob-list">
          {result.group_probs.map((p) => (
            <ProbBar key={p.name} name={p.name} value={p.probability} color={c.fill} />
          ))}
        </div>
      </div>

      {/* Recommendation */}
      <div className="recommendation" style={{ borderLeftColor: c.fill, background: c.bg, color: c.text }}>
        <strong>Ranger Action:</strong> {result.recommendation}
      </div>
    </div>
  );
}

function Fact({ label, value }) {
  return (
    <div className="fact-item">
      <div className="fact-label">{label}</div>
      <div className="fact-value">{value}</div>
    </div>
  );
}

function ProbBar({ name, value, color }) {
  return (
    <div className="prob-row">
      <span className="prob-name">{name}</span>
      <div className="prob-track">
        <div
          className="prob-fill"
          style={{ width: `${value}%`, background: color, boxShadow: `0 0 8px ${color}` }}
        />
      </div>
      <span className="prob-pct">{value.toFixed(1)}%</span>
    </div>
  );
}
