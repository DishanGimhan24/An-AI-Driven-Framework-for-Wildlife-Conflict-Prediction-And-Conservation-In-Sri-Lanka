import { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { MapPin, Calendar, Leaf, Car, AlertTriangle, FileText, List, Search, FileDown } from "lucide-react";
import { HIMASHI_INCIDENTS_API } from "../apiConfig";
import "./HimashiAddIncident.css";

const INITIAL_FORM = {
  province: "",
  district: "",
  villageArea: "",
  roadRailway: "",
  nearestLandmark: "",
  incidentDate: "",
  incidentTime: "",
  dayNight: "",
  animalType: "",
  animalCount: "",
  animalAge: "",
  vehicleType: "",
  direction: "",
  injuryAnimal: "",
  deathAnimal: "",
  injuryHumans: "",
  deathHumans: "",
  description: "",
};

const PROVINCES = [
  "Western",
  "Central",
  "Southern",
  "Northern",
  "Eastern",
  "North Western",
  "North Central",
  "Uva",
  "Sabaragamuwa",
];

const DISTRICTS = [
  "Colombo",
  "Gampaha",
  "Kalutara",
  "Kandy",
  "Matale",
  "Nuwara Eliya",
  "Galle",
  "Matara",
  "Hambantota",
  "Jaffna",
  "Kilinochchi",
  "Mannar",
  "Vavuniya",
  "Mullaitivu",
  "Batticaloa",
  "Ampara",
  "Trincomalee",
  "Kurunegala",
  "Puttalam",
  "Anuradhapura",
  "Polonnaruwa",
  "Badulla",
  "Moneragala",
  "Ratnapura",
  "Kegalle",
];

const ROAD_OPTIONS = [
  "A1 - Colombo to Kandy",
  "A2 - Colombo to Galle",
  "A9 - Kandy to Jaffna",
  "Railway Line",
  "Other",
];

const DIRECTION_OPTIONS = [
  "Colombo to Kandy",
  "Kandy to Colombo",
  "Northbound",
  "Southbound",
  "Eastbound",
  "Westbound",
];

const ANIMAL_TYPES = [
  "Elephant",
  "Leopard",
  "Deer",
  "Wild Boar",
  "Buffalo",
  "Other",
];

const AGE_OPTIONS = [
  "Calf",
  "Juvenile",
  "Adult",
  "Unknown",
];

const VEHICLE_TYPES = [
  "Car",
  "Van",
  "Lorry / Truck",
  "Bus",
  "Motorbike",
  "Train",
  "Other",
];

const getTodayString = () => {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, "0");
  const day = String(today.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

const REQUIRED_MESSAGES = {
  province: "Province is required.",
  district: "District is required.",
  villageArea: "Locality / Area is required.",
  roadRailway: "Roadway / Railway Line is required.",
  incidentDate: "Date is required.",
  animalType: "Animal Type is required.",
  vehicleType: "Vehicle Type is required.",
  injuryAnimal: "Please select injury to animal.",
  deathAnimal: "Please select death of animal.",
  injuryHumans: "Please select injury to humans.",
  deathHumans: "Please select death of humans.",
};

const TIME_PATTERN = /^([01]\d|2[0-3]):([0-5]\d)(:[0-5]\d)?$/;
const MAX_DESCRIPTION_LENGTH = 250;

const toDateInputValue = (value) => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value).slice(0, 10);
  }
  return date.toISOString().slice(0, 10);
};

const mapIncidentToForm = (incident) => ({
  province: incident?.province ?? "",
  district: incident?.district ?? "",
  villageArea: incident?.villageArea ?? incident?.village ?? "",
  roadRailway: incident?.roadRailway ?? incident?.road ?? "",
  nearestLandmark: incident?.nearestLandmark ?? incident?.landmark ?? "",
  incidentDate: toDateInputValue(incident?.incidentDate ?? incident?.date),
  incidentTime: incident?.incidentTime ?? incident?.time ?? "",
  dayNight: incident?.dayNight ?? "",
  animalType: incident?.animalType ?? "",
  animalCount: incident?.animalCount ?? incident?.numberOfAnimals ?? "",
  animalAge: incident?.animalAge ?? incident?.age ?? "",
  vehicleType: incident?.vehicleType ?? "",
  direction: incident?.direction ?? "",
  injuryAnimal: incident?.injuryAnimal ?? "",
  deathAnimal: incident?.deathAnimal ?? "",
  injuryHumans: incident?.injuryHumans ?? incident?.injuryHuman ?? "",
  deathHumans: incident?.deathHumans ?? incident?.deathHuman ?? "",
  description: incident?.description ?? "",
});

const validateOptionalField = (name, value) => {
  if (!value) return "";
  if (name === "animalCount") {
    const count = Number(value);
    if (!Number.isFinite(count) || count <= 0) {
      return "Number of animals must be a positive value.";
    }
  }
  if (name === "incidentTime" && !TIME_PATTERN.test(value)) {
    return "Enter a valid time (HH:MM).";
  }
  if (name === "description" && value.length > MAX_DESCRIPTION_LENGTH) {
    return `Description must be ${MAX_DESCRIPTION_LENGTH} characters or less.`;
  }
  return "";
};

export default function HimashiAddIncident() {
  const location = useLocation();
  const { incidentId } = useParams();
  const [formKey, setFormKey] = useState(0);
  const [loadedIncident, setLoadedIncident] = useState(null);
  const [incidentLoading, setIncidentLoading] = useState(Boolean(incidentId && !location.state?.incident?._id));
  const [incidentLoadError, setIncidentLoadError] = useState("");
  const incidentToEdit = location.state?.incident ?? loadedIncident;
  const editingIncidentId = incidentId || incidentToEdit?._id || "";
  const isEditMode = Boolean(editingIncidentId);
  const [formData, setFormData] = useState(() => mapIncidentToForm(incidentToEdit));
  const [errors, setErrors] = useState({});
  const navigate = useNavigate();
  const todayString = getTodayString();

  useEffect(() => {
    if (!incidentId) {
      setLoadedIncident(null);
      setIncidentLoading(false);
      setIncidentLoadError("");
      return;
    }

    if (location.state?.incident?._id === incidentId) {
      setLoadedIncident(location.state.incident);
      setIncidentLoading(false);
      setIncidentLoadError("");
      return;
    }

    const controller = new AbortController();

    const loadIncident = async () => {
      try {
        setIncidentLoading(true);
        setIncidentLoadError("");

        const response = await fetch(`${HIMASHI_INCIDENTS_API}/${incidentId}`, {
          signal: controller.signal,
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.message || "Failed to load incident.");
        }

        const incident = await response.json();
        setLoadedIncident(incident);
      } catch (error) {
        if (error.name !== "AbortError") {
          setIncidentLoadError(error.message || "Failed to load incident.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setIncidentLoading(false);
        }
      }
    };

    loadIncident();

    return () => controller.abort();
  }, [incidentId, location.state]);

  useEffect(() => {
    setFormData(mapIncidentToForm(incidentToEdit));
    setFormKey((prev) => prev + 1);
    setErrors({});
  }, [incidentToEdit]);

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setErrors((prev) => {
      const next = { ...prev };
      if (name === "incidentDate") {
        if (value && value > todayString) {
          next.incidentDate = "Incident date cannot be a future date.";
          return next;
        }
        delete next.incidentDate;
        return next;
      }

      const optionalMessage = validateOptionalField(name, value);
      if (optionalMessage) {
        next[name] = optionalMessage;
        return next;
      }

      if (REQUIRED_MESSAGES[name]) {
        if (value) {
          delete next[name];
        }
        return next;
      }

      if (!value && next[name]) {
        delete next[name];
      }
      return next;
    });
  };

  const handleSave = async (event) => {
    event.preventDefault();
    const nextErrors = {};
    Object.entries(REQUIRED_MESSAGES).forEach(([field, message]) => {
      if (!formData[field]) {
        nextErrors[field] = message;
      }
    });

    if (formData.incidentDate && formData.incidentDate > todayString) {
      nextErrors.incidentDate = "Incident date cannot be a future date.";
    }

    ["animalCount", "incidentTime", "description"].forEach((field) => {
      const message = validateOptionalField(field, formData[field]);
      if (message) {
        nextErrors[field] = message;
      }
    });

    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) {
      return;
    }
    const payload = {
      province: formData.province,
      district: formData.district,
      villageArea: formData.villageArea,
      roadRailway: formData.roadRailway,
      nearestLandmark: formData.nearestLandmark,
      incidentDate: formData.incidentDate,
      incidentTime: formData.incidentTime,
      dayNight: formData.dayNight,
      animalType: formData.animalType,
      animalCount: Number(formData.animalCount),
      animalAge: formData.animalAge,
      vehicleType: formData.vehicleType,
      direction: formData.direction,
      injuryAnimal: formData.injuryAnimal,
      deathAnimal: formData.deathAnimal,
      injuryHumans: formData.injuryHumans,
      deathHumans: formData.deathHumans,
      description: formData.description,
    };

    try {
      const response = await fetch(
        isEditMode ? `${HIMASHI_INCIDENTS_API}/${editingIncidentId}` : `${HIMASHI_INCIDENTS_API}/add`,
        {
        method: isEditMode ? "PUT" : "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload.message || "Request failed");
      }

      await response.json().catch(() => null);

      if (isEditMode) {
        alert("Incident updated successfully");
        navigate("/incidents");
        return;
      }

      alert("Incident saved successfully");
      handleReset();
    } catch (error) {
      alert(error.message || `Failed to ${isEditMode ? "update" : "save"} incident`);
    }
  };

  const handleReset = () => {
    setFormData(isEditMode ? mapIncidentToForm(incidentToEdit) : { ...INITIAL_FORM });
    setFormKey((prev) => prev + 1);
    setErrors({});
  };

  const handleCancel = () => {
    if (isEditMode) {
      navigate("/incidents");
      return;
    }
    if (window.history.length > 1) {
      navigate(-1);
      return;
    }
    handleReset();
  };

  if (incidentLoading) {
    return (
      <div className="incident-page">
        <div className="incident-layout">
          <div className="incident-card">
            <div className="incident-card__header">
              <div>
                <h1>Edit Incident Report</h1>
                <p>Loading incident details...</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (incidentId && incidentLoadError) {
    return (
      <div className="incident-page">
        <div className="incident-layout">
          <div className="incident-card">
            <div className="incident-card__header">
              <div>
                <h1>Edit Incident Report</h1>
                <p>{incidentLoadError}</p>
              </div>
            </div>
            <div className="incident-actions">
              <button type="button" className="incident-btn incident-btn--secondary" onClick={() => navigate("/incidents")}>
                Back to List
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="incident-page">
      <div className="incident-layout">
        <div className="incident-card">
          <div className="incident-card__header">
            <div>
              <h1>{isEditMode ? "Edit Incident Report" : "Incident Reporting Form"}</h1>
              <p>
                {isEditMode
                  ? "Update an existing wildlife-vehicle collision record"
                  : "Official record for wildlife-vehicle collision incidents"}
              </p>
            </div>
          </div>

          <form key={formKey} className="incident-form" onSubmit={handleSave}>
            <div className="incident-form__note">
              Please complete all mandatory fields marked <span className="incident-required">*</span>.
            </div>

            <section className="incident-section">
            <div className="incident-section__header">
              <span className="incident-section__icon"><MapPin size={16} /></span>
              <h2>1. Location Information</h2>
            </div>
            <div className="incident-grid">
              <div className="incident-field">
                <label htmlFor="province">Province <span className="incident-required">*</span></label>
                <select
                  id="province"
                  name="province"
                  value={formData.province}
                  onChange={handleInputChange}
                >
                  <option value="">Select province</option>
                  {PROVINCES.map((province) => (
                    <option key={province} value={province}>{province}</option>
                  ))}
                </select>
                {errors.province && <div className="incident-error">{errors.province}</div>}
              </div>
              <div className="incident-field">
                <label htmlFor="district">District <span className="incident-required">*</span></label>
                <select
                  id="district"
                  name="district"
                  value={formData.district}
                  onChange={handleInputChange}
                >
                  <option value="">Select district</option>
                  {DISTRICTS.map((district) => (
                    <option key={district} value={district}>{district}</option>
                  ))}
                </select>
                {errors.district && <div className="incident-error">{errors.district}</div>}
              </div>
              <div className="incident-field">
                <label htmlFor="villageArea">Locality / Area <span className="incident-required">*</span></label>
                <input
                  id="villageArea"
                  name="villageArea"
                  type="text"
                  placeholder="Enter locality or area"
                  value={formData.villageArea}
                  onChange={handleInputChange}
                />
                {errors.villageArea && <div className="incident-error">{errors.villageArea}</div>}
              </div>
              <div className="incident-field">
                <label htmlFor="roadRailway">Roadway / Railway Line <span className="incident-required">*</span></label>
                <input
                  id="roadRailway"
                  name="roadRailway"
                  type="text"
                  list="road-options"
                  placeholder="Select or enter road/railway line"
                  value={formData.roadRailway}
                  onChange={handleInputChange}
                />
                <datalist id="road-options">
                  {ROAD_OPTIONS.map((option) => (
                    <option key={option} value={option} />
                  ))}
                </datalist>
                {errors.roadRailway && <div className="incident-error">{errors.roadRailway}</div>}
              </div>
              <div className="incident-field">
                <label htmlFor="nearestLandmark">Nearest Landmark / Milepost</label>
                <input
                  id="nearestLandmark"
                  name="nearestLandmark"
                  type="text"
                  placeholder="Enter nearest landmark or reference"
                  value={formData.nearestLandmark}
                  onChange={handleInputChange}
                />
              </div>
            </div>
          </section>

          <section className="incident-section">
            <div className="incident-section__header">
              <span className="incident-section__icon"><Calendar size={16} /></span>
              <h2>2. Incident Details</h2>
            </div>
            <div className="incident-grid incident-grid--three">
              <div className="incident-field">
                <label htmlFor="incidentDate">Date <span className="incident-required">*</span></label>
                <input
                  id="incidentDate"
                  name="incidentDate"
                  type="date"
                  max={todayString}
                  aria-invalid={Boolean(errors.incidentDate)}
                  aria-describedby={errors.incidentDate ? "incidentDate-error" : undefined}
                  value={formData.incidentDate}
                  onChange={handleInputChange}
                />
                {errors.incidentDate && (
                  <div id="incidentDate-error" role="alert" style={{ color: "#b91c1c", fontSize: "12px" }}>
                    {errors.incidentDate}
                  </div>
                )}
              </div>
              <div className="incident-field">
                <label htmlFor="incidentTime">Time</label>
                <input
                  id="incidentTime"
                  name="incidentTime"
                  type="time"
                  value={formData.incidentTime}
                  onChange={handleInputChange}
                />
                {errors.incidentTime && <div className="incident-error">{errors.incidentTime}</div>}
              </div>
              <div className="incident-field">
                <label htmlFor="dayNight">Time of Day</label>
                <select
                  id="dayNight"
                  name="dayNight"
                  value={formData.dayNight}
                  onChange={handleInputChange}
                >
                  <option value="">Select time of day</option>
                  <option value="Day">Day</option>
                  <option value="Night">Night</option>
                </select>
              </div>
            </div>
          </section>

          <section className="incident-section">
            <div className="incident-section__header">
              <span className="incident-section__icon"><Leaf size={16} /></span>
              <h2>3. Animal Details</h2>
            </div>
            <div className="incident-grid incident-grid--three">
              <div className="incident-field">
                <label htmlFor="animalType">Animal Type <span className="incident-required">*</span></label>
                <select
                  id="animalType"
                  name="animalType"
                  value={formData.animalType}
                  onChange={handleInputChange}
                >
                  <option value="">Select animal</option>
                  {ANIMAL_TYPES.map((animal) => (
                    <option key={animal} value={animal}>{animal}</option>
                  ))}
                </select>
                {errors.animalType && <div className="incident-error">{errors.animalType}</div>}
              </div>
              <div className="incident-field">
                <label htmlFor="animalCount">Number of Animals Involved</label>
                <input
                  id="animalCount"
                  name="animalCount"
                  type="number"
                  min="1"
                  placeholder="Enter count"
                  value={formData.animalCount}
                  onChange={handleInputChange}
                />
                {errors.animalCount && <div className="incident-error">{errors.animalCount}</div>}
              </div>
              <div className="incident-field">
                <label htmlFor="animalAge">Age Category (if known)</label>
                <select
                  id="animalAge"
                  name="animalAge"
                  value={formData.animalAge}
                  onChange={handleInputChange}
                >
                  <option value="">Select category</option>
                  {AGE_OPTIONS.map((age) => (
                    <option key={age} value={age}>{age}</option>
                  ))}
                </select>
              </div>
            </div>
          </section>

          <section className="incident-section">
            <div className="incident-section__header">
              <span className="incident-section__icon"><Car size={16} /></span>
              <h2>4. Vehicle Details</h2>
            </div>
            <div className="incident-grid">
              <div className="incident-field">
                <label htmlFor="vehicleType">Vehicle Type <span className="incident-required">*</span></label>
                <select
                  id="vehicleType"
                  name="vehicleType"
                  value={formData.vehicleType}
                  onChange={handleInputChange}
                >
                  <option value="">Select vehicle</option>
                  {VEHICLE_TYPES.map((vehicle) => (
                    <option key={vehicle} value={vehicle}>{vehicle}</option>
                  ))}
                </select>
                {errors.vehicleType && <div className="incident-error">{errors.vehicleType}</div>}
              </div>
              <div className="incident-field">
                <label htmlFor="direction">Direction of Travel</label>
                <input
                  id="direction"
                  name="direction"
                  type="text"
                  list="direction-options"
                  placeholder="Select or enter direction"
                  value={formData.direction}
                  onChange={handleInputChange}
                />
                <datalist id="direction-options">
                  {DIRECTION_OPTIONS.map((option) => (
                    <option key={option} value={option} />
                  ))}
                </datalist>
              </div>
            </div>
          </section>

          <section className="incident-section">
            <div className="incident-section__header">
              <span className="incident-section__icon"><AlertTriangle size={16} /></span>
              <h2>5. Impact Details</h2>
            </div>
            <div className="incident-grid incident-grid--four">
              <div className="incident-radio-group">
                <span className="incident-radio-title">Injury to Animal <span className="incident-required">*</span></span>
                <div className="incident-radio-options">
                  <label className="incident-radio">
                    <input
                      type="radio"
                      name="injuryAnimal"
                      value="Yes"
                      checked={formData.injuryAnimal === "Yes"}
                      onChange={handleInputChange}
                    />
                    Yes
                  </label>
                  <label className="incident-radio">
                    <input
                      type="radio"
                      name="injuryAnimal"
                      value="No"
                      checked={formData.injuryAnimal === "No"}
                      onChange={handleInputChange}
                    />
                    No
                  </label>
                </div>
                {errors.injuryAnimal && <div className="incident-error">{errors.injuryAnimal}</div>}
              </div>
              <div className="incident-radio-group">
                <span className="incident-radio-title">Death of Animal <span className="incident-required">*</span></span>
                <div className="incident-radio-options">
                  <label className="incident-radio">
                    <input
                      type="radio"
                      name="deathAnimal"
                      value="Yes"
                      checked={formData.deathAnimal === "Yes"}
                      onChange={handleInputChange}
                    />
                    Yes
                  </label>
                  <label className="incident-radio">
                    <input
                      type="radio"
                      name="deathAnimal"
                      value="No"
                      checked={formData.deathAnimal === "No"}
                      onChange={handleInputChange}
                    />
                    No
                  </label>
                </div>
                {errors.deathAnimal && <div className="incident-error">{errors.deathAnimal}</div>}
              </div>
              <div className="incident-radio-group">
                <span className="incident-radio-title">Injury to Humans <span className="incident-required">*</span></span>
                <div className="incident-radio-options">
                  <label className="incident-radio">
                    <input
                      type="radio"
                      name="injuryHumans"
                      value="Yes"
                      checked={formData.injuryHumans === "Yes"}
                      onChange={handleInputChange}
                    />
                    Yes
                  </label>
                  <label className="incident-radio">
                    <input
                      type="radio"
                      name="injuryHumans"
                      value="No"
                      checked={formData.injuryHumans === "No"}
                      onChange={handleInputChange}
                    />
                    No
                  </label>
                </div>
                {errors.injuryHumans && <div className="incident-error">{errors.injuryHumans}</div>}
              </div>
              <div className="incident-radio-group">
                <span className="incident-radio-title">Death of Humans <span className="incident-required">*</span></span>
                <div className="incident-radio-options">
                  <label className="incident-radio">
                    <input
                      type="radio"
                      name="deathHumans"
                      value="Yes"
                      checked={formData.deathHumans === "Yes"}
                      onChange={handleInputChange}
                    />
                    Yes
                  </label>
                  <label className="incident-radio">
                    <input
                      type="radio"
                      name="deathHumans"
                      value="No"
                      checked={formData.deathHumans === "No"}
                      onChange={handleInputChange}
                    />
                    No
                  </label>
                </div>
                {errors.deathHumans && <div className="incident-error">{errors.deathHumans}</div>}
              </div>
            </div>
          </section>

          <section className="incident-section">
            <div className="incident-section__header">
              <span className="incident-section__icon"><FileText size={16} /></span>
              <h2>6. Incident Narrative</h2>
            </div>
            <div className="incident-field">
              <label htmlFor="description">Incident Description</label>
              <textarea
                id="description"
                name="description"
                rows="4"
                placeholder="Provide a clear summary of the incident"
                value={formData.description}
                onChange={handleInputChange}
              />
              {errors.description && <div className="incident-error">{errors.description}</div>}
            </div>
          </section>

          <div className="incident-actions">
            <button type="submit" className="incident-btn incident-btn--primary">{isEditMode ? "Update Incident" : "Save Incident"}</button>
            <button type="button" className="incident-btn incident-btn--secondary" onClick={handleReset}>{isEditMode ? "Reset Changes" : "Reset Form"}</button>
            <button type="button" className="incident-btn incident-btn--secondary" onClick={() => navigate("/incidents")}>{isEditMode ? "Back to List" : "View Data"}</button>
            <button type="button" className="incident-btn incident-btn--danger" onClick={handleCancel}>Cancel</button>
          </div>
          </form>
        </div>

        <aside className="incident-side">
          <div className="incident-side__card incident-side__actions">
            <div className="incident-side__title">Quick Actions</div>
            <button
              type="button"
              className="incident-side__btn incident-side__btn--primary"
              onClick={() => navigate("/incidents")}
            >
              <List size={16} />
              View All Incidents
            </button>
            <button
              type="button"
              className="incident-side__btn incident-side__btn--secondary"
              onClick={() => navigate("/risk-map?focus=search")}
            >
              <Search size={16} />
              Search Incidents
            </button>
            <button
              type="button"
              className="incident-side__btn incident-side__btn--accent"
              onClick={() => navigate("/risk-summary")}
            >
              <FileDown size={16} />
              Generate Report (PDF)
            </button>
          </div>

          <div className="incident-side__card">
            <div className="incident-side__title">Instruction</div>
            <ul className="incident-side__list">
              <li>Please fill all required fields (<span className="incident-required">*</span>).</li>
              <li>Provide accurate location details.</li>
              <li>Use the correct time and date.</li>
              <li>Add a full description of the incident.</li>
              <li>{isEditMode ? "Click Update Incident to save your changes." : "Click Save Incident to record."}</li>
            </ul>
          </div>

          <div className="incident-side__card incident-side__important">
            <div className="incident-side__title">Important</div>
            <p className="incident-side__note">
              Accurate data helps protect wildlife and prevent future collisions. Thank you.
            </p>
            <div className="incident-side__art" aria-hidden="true" />
          </div>
        </aside>
      </div>
    </div>
  );
}
