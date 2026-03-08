import { useEffect, useState } from "react";
import { api } from "../api/client";
import axios from "axios";
import { Loader2 } from "lucide-react";
import { MapContainer, TileLayer, Marker, useMapEvents, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const customIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

const locationMap = {
  // Regions
  "Anuradhapura": [8.3114, 80.4037],
  "Central": [7.2906, 80.6337],
  "Eastern": [7.7170, 81.6989],
  "Head Office-Flying Squad": [6.8970, 79.9223], // Based near Battaramulla/Colombo
  "Kilinochchi": [9.3803, 80.3770],
  "Polonnaruwa": [7.9403, 81.0188],
  "Puttalam": [8.0362, 79.8283],
  "Southern": [6.0535, 80.2210], // General Galle/Southern area
  "Trincomalee": [8.5811, 81.2330],
  "Uwa": [6.9847, 81.0549], // Badulla/Uva province
  "Vauniya": [8.7514, 80.4971], // Vavuniya
  "Wayamba": [7.4818, 80.3609], // Kurunegala/North Western
  "Western": [6.9271, 79.8612], // Colombo

  // Specific Known Locations & Parks
  "Colombo": [6.9271, 79.8612],
  "Galle Face": [6.9238, 79.8458],
  "Kandy": [7.2906, 80.6337],
  "Yala NP": [6.3683, 81.5156],
  "Wilpattu NP": [8.4389, 79.9950],
  "Udawalawe NP": [6.4385, 80.8930],
  "Minneriya NP": [8.0319, 80.8351],
  "Horowpothana NP": [8.6946, 80.8652],
  "Bundala NP": [6.1950, 81.2330],
  "Gal Oya NP": [7.2140, 81.4111],
  "Lunugamwehera NP": [6.3869, 81.1966],
  "Wasgamuwa NP": [7.7126, 80.9329],
  "Angammadilla NP": [7.8288, 80.9576],
  "Kaudulla NP": [8.1818, 80.9258],
  "Somawathiya NP": [8.1396, 81.1970],
  "Hikkaduwa NP": [6.1438, 80.0963],
  "Pigeon Island": [8.7214, 81.1989]
};

// We use Nominatim API to dynamically find coordinates if not in this list

function MapUpdater({ position }) {
  const map = useMap();
  useEffect(() => {
    if (position) {
      map.flyTo(position, 10, { animate: true });
    }
  }, [position, map]);
  return null;
}

function LocationPicker({ position, setPosition }) {
  useMapEvents({
    click(e) {
      setPosition([e.latlng.lat, e.latlng.lng]);
    },
  });
  return position ? <Marker position={position} icon={customIcon} /> : null;
}

export default function Report() {
  const [regions, setRegions] = useState([]);
  const [locations, setLocations] = useState([]);

  const [region, setRegion] = useState("");
  const [location, setLocation] = useState("");

  const [offenceType, setOffenceType] = useState("suspected_poaching");
  const [when, setWhen] = useState("");
  const [description, setDescription] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  useEffect(() => {
    if (!imageFile) {
      setImagePreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(imageFile);
    setImagePreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [imageFile]);

  const [position, setPosition] = useState(null);
  const [isAnonymous, setIsAnonymous] = useState(true);

  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const data = await api.getRegions();
        setRegions(data.regions || []);
        if (data.regions?.length) setRegion(data.regions[0]);
      } catch (e) {
        setError(e.message);
      }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      if (!region) return;
      try {
        const data = await api.getLocations(region);
        setLocations(data.locations || []);
        if (data.locations?.length) setLocation(data.locations[0]);
      } catch (e) {
        setError(e.message);
      }
    })();
  }, [region]);

  useEffect(() => {
    (async () => {
      if (!region || !location) return;

      // 1. Try exact matches in our fast local dictionary
      if (locationMap[location]) {
        setPosition(locationMap[location]);
        return;
      }
      if (locationMap[region]) {
        setPosition(locationMap[region]);
        return;
      }

      // 2. Fallback to dynamically searching OpenStreetMap
      try {
        const query = encodeURIComponent(`${location}, ${region}, Sri Lanka`);
        const res = await axios.get(`https://nominatim.openstreetmap.org/search?format=json&q=${query}&limit=1`);
        if (res.data && res.data.length > 0) {
          setPosition([parseFloat(res.data[0].lat), parseFloat(res.data[0].lon)]);
        } else {
          setPosition([7.8731, 80.7718]); // Center of Sri Lanka fallback
        }
      } catch (e) {
        setPosition([7.8731, 80.7718]);
      }
    })();
  }, [region, location]);

  async function submit(e) {
    if (e) e.preventDefault();
    setError("");
    setResult(null);

    if (!region || !location || !description.trim()) {
      setError("Please fill all required fields: region, location, and description.");
      return;
    }

    setSubmitting(true);
    try {
      let imageUrl = null;
      if (imageFile) {
        const formData = new FormData();
        formData.append("file", imageFile);
        const uploadRes = await axios.post("http://127.0.0.1:8000/api/upload", formData, {
          headers: { "Content-Type": "multipart/form-data" }
        });
        imageUrl = uploadRes.data.image_url;
      }

      const payload = {
        region,
        location,
        offence_type: offenceType,
        incident_datetime: when || null,
        description: description.trim(),
        image_url: imageUrl,
        latitude: position ? position[0] : null,
        longitude: position ? position[1] : null,
        is_anonymous: isAnonymous,
      };

      const response = await axios.post("http://127.0.0.1:8000/api/reports", payload);
      setResult(response.data);

      // Reset form on success
      setRegion(regions.length > 0 ? regions[0] : "");
      setLocation(locations.length > 0 ? locations[0] : "");
      setOffenceType("suspected_poaching");
      setWhen("");
      setDescription("");
      setImageFile(null);
      setPosition(null);

    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="report-container">
      <div className="report-header">
        <h1>📝 Community Report</h1>
        <p>Report illegal poaching or wildlife offences. Your report goes to wildlife officers.</p>
      </div>

      {error && <div className="error" style={{ marginBottom: "1rem" }}>{error}</div>}
      {result && (
        <div className="alert-success" style={{ marginBottom: "1rem" }}>
          <div className="alert-success-title">✓ Report submitted successfully!</div>
          <p>Reference ID: <strong style={{ fontSize: "1.1rem" }}>{result.reference_id}</strong></p>
          <p>Status: <span style={{ textTransform: "capitalize" }}>{result.status}</span></p>
        </div>
      )}

      <div className="glass-card">
        <form className="report-form" onSubmit={submit}>
          <div className="form-group">
            <label>🌍 Region <span style={{ color: "red" }}>*</span></label>
            <select required value={region} onChange={(e) => setRegion(e.target.value)} className="glass-input">
              {regions.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>

          <div className="form-group">
            <label>📍 General Location <span style={{ color: "red" }}>*</span></label>
            <select required value={location} onChange={(e) => setLocation(e.target.value)} className="glass-input">
              {locations.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>

          <div className="form-group">
            <label>📌 Exact Location on Map (Optional)</label>
            <p style={{ fontSize: "0.85rem", color: "#aaa", marginBottom: "8px" }}>Click on the map to place a pin at the exact incident location.</p>
            <div style={{ height: "300px", width: "100%", borderRadius: "8px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.1)" }}>
              <MapContainer center={[7.8731, 80.7718]} zoom={7} style={{ height: "100%", width: "100%", zIndex: 1 }}>
                <TileLayer
                  url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />
                <MapUpdater position={position} />
                <LocationPicker position={position} setPosition={setPosition} />
              </MapContainer>
            </div>
            {position && (
              <small style={{ color: "#22c55e", display: "flex", justifyContent: "space-between", marginTop: "4px" }}>
                <span>Pin placed successfully!</span>
                <span style={{ cursor: "pointer", color: "#ef4444" }} onClick={() => setPosition(null)}>Remove Pin</span>
              </small>
            )}
          </div>

          <div className="form-group">
            <label>⚠️ Offence type (what you suspect)</label>
            <select value={offenceType} onChange={(e) => setOffenceType(e.target.value)} className="glass-input">
              <option value="suspected_poaching">Suspected poaching</option>
              <option value="illegal_logging">Illegal logging</option>
              <option value="traps_or_weapons">Traps or weapons</option>
              <option value="meat_trade">Meat or egg trade</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="form-group">
            <label>📅 Date & time (optional)</label>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <input
                type="datetime-local"
                value={when}
                onChange={(e) => setWhen(e.target.value)}
                className="glass-input"
                style={{ flex: 1 }}
              />
              <button
                type="button"
                className="now-btn"
                onClick={() => {
                  const now = new Date();
                  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000)
                    .toISOString()
                    .slice(0, 16);
                  setWhen(local);
                }}
                title="Use current date & time"
              >
                Now
              </button>
            </div>
          </div>

          <div className="form-group">
            <label>📄 Description <span style={{ color: "red" }}>*</span></label>
            <textarea
              required
              rows={5}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What happened? Any vehicle number, people count, direction, sounds, or evidence?"
              className="glass-input"
            />
          </div>

          <div className="form-group">
            <label>📸 Photo Evidence (optional)</label>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setImageFile(e.target.files[0] || null)}
                className="glass-input"
                style={{ padding: "0.5rem", display: imageFile ? 'none' : 'block' }}
                id="evidence-upload"
              />

              {imagePreview && (
                <div style={{ position: "relative", width: "fit-content", marginTop: "0.5rem" }}>
                  <img
                    src={imagePreview}
                    alt="Evidence Preview"
                    style={{ width: "100%", maxWidth: "300px", borderRadius: "12px", border: "2px solid rgba(255,255,255,0.1)", objectFit: "cover", boxShadow: "0 4px 15px rgba(0,0,0,0.3)" }}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      setImageFile(null);
                      const input = document.getElementById("evidence-upload");
                      if (input) input.value = "";
                    }}
                    style={{ position: "absolute", top: "-10px", right: "-10px", background: "#ef4444", color: "white", border: "2px solid #222", borderRadius: "50%", width: "26px", height: "26px", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", boxShadow: "0 2px 8px rgba(0,0,0,0.5)", fontWeight: "bold" }}
                    title="Remove selected photo"
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>
          </div>

          <button type="submit" className="submit-btn" disabled={submitting}>
            {submitting ? (
              <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
                <Loader2 className="spin" size={18} style={{ animation: "spin 1s linear infinite" }} />
                Submitting...
              </span>
            ) : "Submit Report"}
          </button>
        </form>
      </div>
    </div>
  );
}
