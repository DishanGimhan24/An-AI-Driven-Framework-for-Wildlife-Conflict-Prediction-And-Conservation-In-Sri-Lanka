import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bus, Car, HeartPulse, Moon, PawPrint, Pencil, Search, Skull, Sun, Train, Trash2 } from "lucide-react";
import { HIMASHI_INCIDENTS_API } from "../apiConfig";
import "./HimashiIncidentList.css";

const formatDate = (value) => {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
};

const isMissingValue = (value) => value === null || value === undefined || value === "";

const pickIncidentField = (incident, keys) => {
  for (const key of keys) {
    const value = incident?.[key];
    if (!isMissingValue(value)) {
      return value;
    }
  }
  return "";
};

const normalizeIncident = (incident = {}) => ({
  ...incident,
  incidentDate: pickIncidentField(incident, ["incidentDate", "date"]),
  incidentTime: pickIncidentField(incident, ["incidentTime", "time"]),
  villageArea: pickIncidentField(incident, ["villageArea", "village"]),
  roadRailway: pickIncidentField(incident, ["roadRailway", "road"]),
  nearestLandmark: pickIncidentField(incident, ["nearestLandmark", "landmark"]),
  animalCount: pickIncidentField(incident, ["animalCount", "numberOfAnimals"]),
  animalAge: pickIncidentField(incident, ["animalAge", "age"]),
  injuryHumans: pickIncidentField(incident, ["injuryHumans", "injuryHuman"]),
  deathHumans: pickIncidentField(incident, ["deathHumans", "deathHuman"]),
});

const formatValue = (value, fallback = "Not Available") => {
  if (isMissingValue(value) || value === "-") return fallback;
  return value;
};

const normalizeValue = (value) => String(value ?? "").trim();

const renderText = (value, fallback = "Not Available") => {
  const missing = isMissingValue(value) || value === "-";
  const formatted = formatValue(value, fallback);
  return (
    <span className={missing ? "incident-cell--muted" : undefined}>
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
    return (
      <span className="incident-pill incident-pill--day">
        <Sun size={12} />
        Day
      </span>
    );
  }
  if (normalized === "night") {
    return (
      <span className="incident-pill incident-pill--night">
        <Moon size={12} />
        Night
      </span>
    );
  }
  return <span className="incident-pill incident-pill--neutral">{value}</span>;
};

const renderAnimalBadge = (value) => {
  const label = formatValue(value);
  const missing = isMissingValue(value) || value === "-";
  if (missing) {
    return <span className="incident-tag incident-tag--muted">{label}</span>;
  }
  return (
    <span className="incident-tag">
      <PawPrint size={14} />
      {label}
    </span>
  );
};

const getVehicleIcon = (value) => {
  const normalized = normalizeValue(value).toLowerCase();
  if (normalized.includes("train") || normalized.includes("rail")) return Train;
  if (normalized.includes("bus")) return Bus;
  if (normalized.includes("car") || normalized.includes("van") || normalized.includes("truck") || normalized.includes("lorry")) return Car;
  return Car;
};

const renderVehicleBadge = (value) => {
  const label = formatValue(value);
  const missing = isMissingValue(value) || value === "-";
  if (missing) {
    return <span className="incident-tag incident-tag--muted">{label}</span>;
  }
  const Icon = getVehicleIcon(value);
  return (
    <span className="incident-tag">
      <Icon size={14} />
      {label}
    </span>
  );
};

const isYesValue = (value) => {
  const normalized = normalizeValue(value).toLowerCase();
  return ["yes", "y", "true", "1"].includes(normalized);
};

const hasInjury = (incident) => (
  isYesValue(incident.injuryAnimal) || isYesValue(incident.injuryHumans)
);

const hasFatal = (incident) => (
  isYesValue(incident.deathAnimal) || isYesValue(incident.deathHumans)
);

const matchesQuery = (incident, rawQuery) => {
  const query = normalizeValue(rawQuery).toLowerCase();
  if (!query) return true;
  const searchable = [
    incident.province,
    incident.district,
    incident.villageArea,
    incident.roadRailway,
    incident.nearestLandmark,
    incident.dayNight,
    incident.animalType,
    incident.vehicleType,
    incident.description,
    incident.incidentTime,
    formatDate(incident.incidentDate),
  ]
    .map((value) => normalizeValue(value).toLowerCase())
    .filter(Boolean)
    .join(" ");
  return searchable.includes(query);
};

const CSV_HEADERS = [
  "No",
  "Date",
  "Time",
  "Province",
  "District",
  "Village/Area",
  "Road/Railway Line",
  "Day/Night",
  "Animal Type",
  "No. of Animals",
  "Vehicle Type",
  "Injury to Animal",
  "Death",
  "Injury to Human",
  "Human Death",
  "Description",
];

const escapeCsv = (value) => {
  const text = String(value ?? "");
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
};

export default function HimashiViewIncidents() {
  const navigate = useNavigate();
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [dayFilter, setDayFilter] = useState("all");
  const [showInjury, setShowInjury] = useState(false);
  const [showFatal, setShowFatal] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    const loadIncidents = async () => {
      try {
        setLoading(true);
        setError("");
        const response = await fetch(HIMASHI_INCIDENTS_API, {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("Failed to load incidents");
        }

        const data = await response.json();
        setIncidents(Array.isArray(data) ? data.map(normalizeIncident) : []);
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

  const filteredIncidents = useMemo(() => {
    return incidents.filter((incident) => {
      const dayNight = normalizeValue(incident.dayNight).toLowerCase();
      if (dayFilter !== "all" && dayNight !== dayFilter) return false;

      if (showInjury || showFatal) {
        const matchFlag = (showInjury && hasInjury(incident)) || (showFatal && hasFatal(incident));
        if (!matchFlag) return false;
      }

      return matchesQuery(incident, query);
    });
  }, [incidents, dayFilter, showInjury, showFatal, query]);

  const emptyMessage = incidents.length === 0
    ? "No incidents found."
    : "No incidents match the current filters.";

  const handleExportCsv = () => {
    const rows = filteredIncidents.map((incident, index) => ([
      index + 1,
      formatDate(incident.incidentDate),
      formatValue(incident.incidentTime, ""),
      formatValue(incident.province),
      formatValue(incident.district),
      formatValue(incident.villageArea),
      formatValue(incident.roadRailway),
      formatValue(incident.dayNight),
      formatValue(incident.animalType),
      formatValue(incident.animalCount),
      formatValue(incident.vehicleType),
      formatValue(incident.injuryAnimal),
      formatValue(incident.deathAnimal),
      formatValue(incident.injuryHumans),
      formatValue(incident.deathHumans),
      formatValue(incident.description),
    ]));

    const csvContent = [CSV_HEADERS, ...rows]
      .map((row) => row.map(escapeCsv).join(","))
      .join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const dateStamp = new Date().toISOString().slice(0, 10);
    link.href = url;
    link.download = `incident-report-${dateStamp}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const handleDeleteIncident = async (incidentId) => {
    if (!incidentId) {
      alert("Unable to delete: missing incident id.");
      return;
    }
    const confirmed = window.confirm("Are you sure you want to delete this incident?");
    if (!confirmed) return;

    try {
      const response = await fetch(`${HIMASHI_INCIDENTS_API}/${incidentId}`,
        { method: "DELETE" }
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const message = payload.message || "Failed to delete incident.";
        throw new Error(message);
      }
      setIncidents((prev) => prev.filter((incident) => incident._id !== incidentId));
    } catch (err) {
      alert(err.message || "Failed to delete incident.");
    }
  };

  const handleEditIncident = (incident) => {
    if (!incident?._id) {
      alert("Unable to edit: missing incident id.");
      return;
    }

    navigate(`/incidents/${incident._id}/edit`, { state: { incident } });
  };

  return (
    <div className="incident-table-page">
      <div className="incident-table-card">
        <div className="incident-table-header">
          <div>
            <h1>Incident Records</h1>
            <p>Latest wildlife-vehicle collision reports</p>
          </div>
          <span className="incident-table-count">
            {filteredIncidents.length} records
            {filteredIncidents.length !== incidents.length && (
              <span className="incident-table-count-sub">of {incidents.length}</span>
            )}
          </span>
        </div>

        {loading && (
          <div className="incident-table-state">Loading incidents...</div>
        )}

        {!loading && error && (
          <div className="incident-table-state incident-table-state--error">{error}</div>
        )}

        {!loading && !error && (
          <div className="incident-table-content">
            <div className="incident-table-toolbar">
              <div className="incident-search" role="search">
                <Search size={16} />
                <input
                  type="search"
                  placeholder="Search by district, animal, vehicle, or notes..."
                  aria-label="Search incidents"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </div>
              <button
                className="incident-toolbar-button"
                type="button"
                onClick={handleExportCsv}
              >
                Generate Report
              </button>
              <div className="incident-filter-group">
                <button
                  type="button"
                  className={`incident-chip ${dayFilter === "all" ? "is-active" : ""}`}
                  onClick={() => setDayFilter("all")}
                >
                  All
                </button>
                <button
                  type="button"
                  className={`incident-chip ${dayFilter === "day" ? "is-active" : ""}`}
                  onClick={() => setDayFilter("day")}
                >
                  <Sun size={14} />
                  Day
                </button>
                <button
                  type="button"
                  className={`incident-chip ${dayFilter === "night" ? "is-active" : ""}`}
                  onClick={() => setDayFilter("night")}
                >
                  <Moon size={14} />
                  Night
                </button>
                <button
                  type="button"
                  className={`incident-chip ${showInjury ? "is-active" : ""}`}
                  onClick={() => setShowInjury((prev) => !prev)}
                >
                  <HeartPulse size={14} />
                  Injury
                </button>
                <button
                  type="button"
                  className={`incident-chip ${showFatal ? "is-active" : ""}`}
                  onClick={() => setShowFatal((prev) => !prev)}
                >
                  <Skull size={14} />
                  Fatal
                </button>
              </div>
            </div>

            <div className="incident-table-body">
            <div className="incident-table-wrapper">
              <table className="incident-table">
                <thead>
                  <tr>
                    <th className="incident-table-no">No</th>
                    <th className="incident-table-date">Date</th>
                    <th className="incident-table-time">Time</th>
                    <th>Province</th>
                    <th>District</th>
                    <th>Village/Area</th>
                    <th>Road/Railway Line</th>
                    <th className="incident-table-flag">Day/Night</th>
                    <th>Animal Type</th>
                    <th className="incident-table-number">No. of Animals</th>
                    <th>Vehicle Type</th>
                    <th className="incident-table-flag">Injury to Animal</th>
                    <th className="incident-table-flag">Death</th>
                    <th className="incident-table-flag">Injury to Human</th>
                    <th className="incident-table-flag">Human Death</th>
                    <th>Description</th>
                    <th className="incident-table-actions">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredIncidents.length === 0 && (
                    <tr>
                      <td colSpan={17} className="incident-table-empty">
                        {emptyMessage}
                      </td>
                    </tr>
                  )}
                  {filteredIncidents.map((incident, index) => {
                    const rowClass = hasFatal(incident)
                      ? "incident-row--fatal"
                      : hasInjury(incident)
                        ? "incident-row--injury"
                        : "";
                    return (
                    <tr
                      key={incident._id || `${incident.incidentDate}-${incident.incidentTime}`}
                      className={rowClass}
                    >
                      <td className="incident-table-no">{index + 1}</td>
                      <td className="incident-table-date">{renderText(formatDate(incident.incidentDate), "-")}</td>
                      <td className="incident-table-time">{renderText(incident.incidentTime, "-")}</td>
                      <td>{renderText(incident.province)}</td>
                      <td>{renderText(incident.district)}</td>
                      <td>{renderText(incident.villageArea)}</td>
                      <td>{renderText(incident.roadRailway)}</td>
                      <td className="incident-table-flag">{renderDayNightBadge(incident.dayNight)}</td>
                      <td>{renderAnimalBadge(incident.animalType)}</td>
                      <td className="incident-table-number">{renderText(incident.animalCount)}</td>
                      <td>{renderVehicleBadge(incident.vehicleType)}</td>
                      <td className="incident-table-flag">{renderBooleanBadge(incident.injuryAnimal)}</td>
                      <td className="incident-table-flag">{renderBooleanBadge(incident.deathAnimal)}</td>
                      <td className="incident-table-flag">{renderBooleanBadge(incident.injuryHumans)}</td>
                      <td className="incident-table-flag">{renderBooleanBadge(incident.deathHumans)}</td>
                      <td
                        className="incident-table-description"
                        title={formatValue(incident.description) === "Not Available" ? "" : formatValue(incident.description)}
                      >
                        {renderText(incident.description)}
                      </td>
                      <td className="incident-table-actions">
                        <div className="incident-action-buttons">
                          <button
                            type="button"
                            className="incident-action-btn incident-action-btn--edit"
                            aria-label="Edit incident"
                            onClick={() => handleEditIncident(incident)}
                          >
                            <Pencil size={14} />
                          </button>
                          <button
                            type="button"
                            className="incident-action-btn incident-action-btn--delete"
                            aria-label="Delete incident"
                            onClick={() => handleDeleteIncident(incident._id)}
                          >
                            <Trash2 size={14} />
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
        )}
      </div>
    </div>
  );
}
