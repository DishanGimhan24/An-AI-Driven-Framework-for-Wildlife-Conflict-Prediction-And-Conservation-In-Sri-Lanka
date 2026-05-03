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

export default function HimashiIncidentList() {
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
                  <th>Date</th>
                  <th>Time</th>
                  <th>Day/Night</th>
                  <th>Province</th>
                  <th>District</th>
                  <th>Village</th>
                  <th>Road</th>
                  <th>Landmark</th>
                  <th>Animal</th>
                  <th>Count</th>
                  <th>Age</th>
                  <th>Vehicle</th>
                  <th>Direction</th>
                  <th>Injury (Animal)</th>
                  <th>Death (Animal)</th>
                  <th>Injury (Human)</th>
                  <th>Death (Human)</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {incidents.length === 0 && (
                  <tr>
                    <td colSpan={18} className="incident-table-empty">
                      No incidents found.
                    </td>
                  </tr>
                )}
                {incidents.map((incident) => (
                  <tr key={incident._id || `${incident.date}-${incident.time}`}>
                    <td>{formatDate(incident.date)}</td>
                    <td>{formatValue(incident.time)}</td>
                    <td>{formatValue(incident.dayNight)}</td>
                    <td>{formatValue(incident.province)}</td>
                    <td>{formatValue(incident.district)}</td>
                    <td>{formatValue(incident.village)}</td>
                    <td>{formatValue(incident.road)}</td>
                    <td>{formatValue(incident.landmark)}</td>
                    <td>{formatValue(incident.animalType)}</td>
                    <td>{formatValue(incident.numberOfAnimals)}</td>
                    <td>{formatValue(incident.age)}</td>
                    <td>{formatValue(incident.vehicleType)}</td>
                    <td>{formatValue(incident.direction)}</td>
                    <td>{formatValue(incident.injuryAnimal)}</td>
                    <td>{formatValue(incident.deathAnimal)}</td>
                    <td>{formatValue(incident.injuryHuman)}</td>
                    <td>{formatValue(incident.deathHuman)}</td>
                    <td className="incident-table-description">
                      {formatValue(incident.description)}
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
