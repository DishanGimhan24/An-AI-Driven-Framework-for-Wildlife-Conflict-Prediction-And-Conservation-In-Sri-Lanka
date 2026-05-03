import { useEffect, useMemo, useState } from "react";
import { Bus, Car, HeartPulse, Moon, PawPrint, Search, Skull, Sun, Train } from "lucide-react";
import "./HimashiIncidentList.css";

const formatDate = (value) => {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
};

const isMissingValue = (value) => value === null || value === undefined || value === "";

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
  isYesValue(incident.injuryAnimal) || isYesValue(incident.injuryHuman)
);

const hasFatal = (incident) => (
  isYesValue(incident.deathAnimal) || isYesValue(incident.deathHuman)
);

const matchesQuery = (incident, rawQuery) => {
  const query = normalizeValue(rawQuery).toLowerCase();
  if (!query) return true;
  const searchable = [
    incident.province,
    incident.district,
    incident.village,
    incident.road,
    incident.dayNight,
    incident.animalType,
    incident.vehicleType,
    incident.description,
    incident.time,
    formatDate(incident.date),
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
      formatDate(incident.date),
      formatValue(incident.time, ""),
      formatValue(incident.province),
      formatValue(incident.district),
      formatValue(incident.village),
      formatValue(incident.road),
      formatValue(incident.dayNight),
      formatValue(incident.animalType),
      formatValue(incident.numberOfAnimals),
      formatValue(incident.vehicleType),
      formatValue(incident.injuryAnimal),
      formatValue(incident.deathAnimal),
      formatValue(incident.injuryHuman),
      formatValue(incident.deathHuman),
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
                  </tr>
                </thead>
                <tbody>
                  {filteredIncidents.length === 0 && (
                    <tr>
                      <td colSpan={16} className="incident-table-empty">
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
                      key={incident._id || `${incident.date}-${incident.time}`}
                      className={rowClass}
                    >
                      <td className="incident-table-no">{index + 1}</td>
                      <td className="incident-table-date">{renderText(formatDate(incident.date), "-")}</td>
                      <td className="incident-table-time">{renderText(incident.time, "-")}</td>
                      <td>{renderText(incident.province)}</td>
                      <td>{renderText(incident.district)}</td>
                      <td>{renderText(incident.village)}</td>
                      <td>{renderText(incident.road)}</td>
                      <td className="incident-table-flag">{renderDayNightBadge(incident.dayNight)}</td>
                      <td>{renderAnimalBadge(incident.animalType)}</td>
                      <td className="incident-table-number">{renderText(incident.numberOfAnimals)}</td>
                      <td>{renderVehicleBadge(incident.vehicleType)}</td>
                      <td className="incident-table-flag">{renderBooleanBadge(incident.injuryAnimal)}</td>
                      <td className="incident-table-flag">{renderBooleanBadge(incident.deathAnimal)}</td>
                      <td className="incident-table-flag">{renderBooleanBadge(incident.injuryHuman)}</td>
                      <td className="incident-table-flag">{renderBooleanBadge(incident.deathHuman)}</td>
                      <td
                        className="incident-table-description"
                        title={formatValue(incident.description) === "Not Available" ? "" : formatValue(incident.description)}
                      >
                        {renderText(incident.description)}
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
