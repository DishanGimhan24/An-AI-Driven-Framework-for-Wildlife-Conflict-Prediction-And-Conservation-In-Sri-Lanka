import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { MapPin, Calendar, Leaf, Car, AlertTriangle, FileText, List, Search, FileDown } from "lucide-react";
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

export default function HimashiAddIncident() {
  const [formData, setFormData] = useState({ ...INITIAL_FORM });
  const [formKey, setFormKey] = useState(0);
  const [dateError, setDateError] = useState("");
  const navigate = useNavigate();
  const todayString = getTodayString();

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    if (name === "incidentDate") {
      const error = value && value > todayString
        ? "Incident date cannot be a future date."
        : "";
      setDateError(error);
    }
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSave = async (event) => {
    event.preventDefault();
    if (formData.incidentDate && formData.incidentDate > todayString) {
      setDateError("Incident date cannot be a future date.");
      return;
    }
    const payload = {
      province: formData.province,
      district: formData.district,
      village: formData.villageArea,
      road: formData.roadRailway,
      landmark: formData.nearestLandmark,
      date: formData.incidentDate,
      time: formData.incidentTime,
      dayNight: formData.dayNight,
      animalType: formData.animalType,
      numberOfAnimals: formData.animalCount,
      age: formData.animalAge,
      vehicleType: formData.vehicleType,
      direction: formData.direction,
      injuryAnimal: formData.injuryAnimal,
      deathAnimal: formData.deathAnimal,
      injuryHuman: formData.injuryHumans,
      deathHuman: formData.deathHumans,
      description: formData.description,
    };

    try {
      const response = await fetch("http://localhost:5000/api/incidents/add", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Request failed");
      }

      alert("Incident saved successfully");
      handleReset();
    } catch (error) {
      alert("Failed to save incident");
    }
  };

  const handleReset = () => {
    setFormData({ ...INITIAL_FORM });
    setFormKey((prev) => prev + 1);
    setDateError("");
  };

  const handleCancel = () => {
    if (window.history.length > 1) {
      navigate(-1);
      return;
    }
    handleReset();
  };

  return (
    <div className="incident-page">
      <div className="incident-layout">
        <div className="incident-card">
          <div className="incident-card__header">
            <div>
              <h1>Incident Reporting Form</h1>
              <p>Official record for wildlife-vehicle collision incidents</p>
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
              </div>
              <div className="incident-field">
                <label htmlFor="nearestLandmark">Nearest Landmark / Milepost <span className="incident-required">*</span></label>
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
                  aria-invalid={Boolean(dateError)}
                  aria-describedby={dateError ? "incidentDate-error" : undefined}
                  value={formData.incidentDate}
                  onChange={handleInputChange}
                />
                {dateError && (
                  <div id="incidentDate-error" role="alert" style={{ color: "#b91c1c", fontSize: "12px" }}>
                    {dateError}
                  </div>
                )}
              </div>
              <div className="incident-field">
                <label htmlFor="incidentTime">Time <span className="incident-required">*</span></label>
                <input
                  id="incidentTime"
                  name="incidentTime"
                  type="time"
                  value={formData.incidentTime}
                  onChange={handleInputChange}
                />
              </div>
              <div className="incident-field">
                <label htmlFor="dayNight">Time of Day <span className="incident-required">*</span></label>
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
              </div>
              <div className="incident-field">
                <label htmlFor="animalCount">Number of Animals Involved <span className="incident-required">*</span></label>
                <input
                  id="animalCount"
                  name="animalCount"
                  type="number"
                  min="1"
                  placeholder="Enter count"
                  value={formData.animalCount}
                  onChange={handleInputChange}
                />
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
              </div>
              <div className="incident-field">
                <label htmlFor="direction">Direction of Travel <span className="incident-required">*</span></label>
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
              </div>
            </div>
          </section>

          <section className="incident-section">
            <div className="incident-section__header">
              <span className="incident-section__icon"><FileText size={16} /></span>
              <h2>6. Incident Narrative</h2>
            </div>
            <div className="incident-field">
              <label htmlFor="description">Incident Description <span className="incident-required">*</span></label>
              <textarea
                id="description"
                name="description"
                rows="4"
                placeholder="Provide a clear summary of the incident"
                value={formData.description}
                onChange={handleInputChange}
              />
            </div>
          </section>

          <div className="incident-actions">
            <button type="submit" className="incident-btn incident-btn--primary">Save Incident</button>
            <button type="button" className="incident-btn incident-btn--secondary" onClick={handleReset}>Reset Form</button>
            <button type="button" className="incident-btn incident-btn--secondary" onClick={() => navigate("/incidents")}>View Data</button>
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
              <li>Click Save Incident to record.</li>
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
