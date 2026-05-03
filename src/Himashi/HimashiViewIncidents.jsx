import { useEffect, useState } from "react";
import "./HimashiIncidentList.css";

const formatDate = (value) => {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
};

const formatValue = (value) => {
  if (value === null || value === undefined || value === "") return "-";
  return value;
};

const normalizeValue = (value) => String(value ?? "").trim();

const renderText = (value) => {
  const formatted = formatValue(value);
  const isEmpty = formatted === "-";
  return (
    <span className={isEmpty ? "incident-cell--muted" : undefined}>
      {formatted}
    </span>
  );
};

const renderBooleanBadge = (value) => {
  const normalized = normalizeValue(value).toLowerCase();
  if (!normalized) {
    return <span className="incident-pill incident-pill--neutral">-</span>;
  }
  if (["yes", "y", "true", "1"].includes(normalized)) {
    return <span className="incident-pill incident-pill--yes">Yes</span>;
  }
  if (["no", "n", "false", "0"].includes(normalized)) {
    return <span className="incident-pill incident-pill--no">No</span>;
  }
  return <span className="incident-pill incident-pill--neutral">{value}</span>;
};

const renderDayNightBadge = (value) => {
  const normalized = normalizeValue(value).toLowerCase();
  if (!normalized) {
    return <span className="incident-pill incident-pill--neutral">-</span>;
  }
  if (normalized === "day") {
    return <span className="incident-pill incident-pill--day">Day</span>;
  }
  if (normalized === "night") {
    return <span className="incident-pill incident-pill--night">Night</span>;
  }
  return <span className="incident-pill incident-pill--neutral">{value}</span>;
};

export default function HimashiViewIncidents() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    const loadIncidents = async () => {
      try {
        setLoading(true);
        setError("");
        const response = await fetch("http://localhost:5000/api/incidents", {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("Failed to load incidents");
        }

        const data = await response.json();
        setIncidents(Array.isArray(data) ? data : []);
      } catch (err) {
        if (err.name !== "AbortError") {
          setError("Failed to load incidents.");
        }
      } finally {
        setLoading(false);
      }
    };

    loadIncidents();
    return () => controller.abort();
  }, []);

  return (
    <div className="incident-table-page">
      <div className="incident-table-card">
        <div className="incident-table-header">
          <div>
            <h1>Incident Records</h1>
            <p>Latest wildlife-vehicle collision reports</p>
          </div>
          <span className="incident-table-count">{incidents.length} records</span>
        </div>

        {loading && (
          <div className="incident-table-state">Loading incidents...</div>
        )}

        {!loading && error && (
          <div className="incident-table-state incident-table-state--error">{error}</div>
        )}

        {!loading && !error && (
          <div className="incident-table-wrapper">
            <table className="incident-table">
              <thead>
                <tr>
                  <th>Province</th>
                  <th>District</th>
                  <th>Village/Area</th>
                  <th>Road/Railway Line</th>
                  <th className="incident-table-date">Date</th>
                  <th className="incident-table-time">Time</th>
                  <th className="incident-table-flag">Day/Night</th>
                  <th>Animal Type</th>
                  <th className="incident-table-number">Number of Animals</th>
                  <th>Vehicle Type</th>
                  <th className="incident-table-flag">Injury to Animal</th>
                  <th className="incident-table-flag">Death</th>
                  <th className="incident-table-flag">Injury to Human</th>
                  <th className="incident-table-flag">Human Death</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {incidents.length === 0 && (
                  <tr>
                    <td colSpan={15} className="incident-table-empty">
                      No incidents found.
                    </td>
                  </tr>
                )}
                {incidents.map((incident) => (
                  <tr key={incident._id || `${incident.date}-${incident.time}`}>
                    <td>{renderText(incident.province)}</td>
                    <td>{renderText(incident.district)}</td>
                    <td>{renderText(incident.village)}</td>
                    <td>{renderText(incident.road)}</td>
                    <td className="incident-table-date">{renderText(formatDate(incident.date))}</td>
                    <td className="incident-table-time">{renderText(incident.time)}</td>
                    <td className="incident-table-flag">{renderDayNightBadge(incident.dayNight)}</td>
                    <td>{renderText(incident.animalType)}</td>
                    <td className="incident-table-number">{renderText(incident.numberOfAnimals)}</td>
                    <td>{renderText(incident.vehicleType)}</td>
                    <td className="incident-table-flag">{renderBooleanBadge(incident.injuryAnimal)}</td>
                    <td className="incident-table-flag">{renderBooleanBadge(incident.deathAnimal)}</td>
                    <td className="incident-table-flag">{renderBooleanBadge(incident.injuryHuman)}</td>
                    <td className="incident-table-flag">{renderBooleanBadge(incident.deathHuman)}</td>
                    <td
                      className="incident-table-description"
                      title={formatValue(incident.description) === "-" ? "" : formatValue(incident.description)}
                    >
                      {renderText(incident.description)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
