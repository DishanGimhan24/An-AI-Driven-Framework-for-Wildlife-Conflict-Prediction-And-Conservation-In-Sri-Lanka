import { useState } from "react";
import "./Prediction.css";

export default function Prediction() {
  const [district, setDistrict] = useState("");
  const [result, setResult] = useState(null);

  const [form, setForm] = useState({
    rain: 50,
    NDVI: 0.5,
    water_distance: 1000,
    distance_to_forest: 800
  });

  // Known elephant areas
  const elephantDistricts = [
    "Anuradhapura",
    "Polonnaruwa",
    "Mullativu",
    "Monaragala",
    "Hambantota",
    "Vavuniyawa",
    "Ampara",
    "Badulla"
  ];

  const handlePredict = async () => {
    if (!elephantDistricts.includes(district)) {
      setResult({
        message: "Outside known elephant zones – risk estimation not applicable"
      });
      return;
    }

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rain: Number(form.rain),
        NDVI: Number(form.NDVI),
        water_distance: Number(form.water_distance),
        distance_to_forest: Number(form.distance_to_forest),
        animal_type: 1,
        vehicle_type: 1
      })
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="prediction-page">
      <div className="prediction-card">
        <h2 className="title">Risk Prediction</h2>

        <label>District</label>
        <select
          value={district}
          onChange={(e) => setDistrict(e.target.value)}
        >
          <option value="">Select district</option>
          <option>Galle</option>
          <option>Colombo</option>
          <option>Anuradhapura</option>
          <option>Monaragala</option>
          <option>Hambantota</option>
          <option>Mullativu</option>
          <option>Vavuniyawa</option>
        </select>

        <label>Rainfall (mm)</label>
        <input
          type="number"
          value={form.rain}
          onChange={(e) => setForm({ ...form, rain: e.target.value })}
        />

        <label>NDVI</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={form.NDVI}
          onChange={(e) => setForm({ ...form, NDVI: e.target.value })}
        />
        <small className="ndvi-value">Value: {form.NDVI}</small>

        <label>Distance to Water (m)</label>
        <input
          type="number"
          value={form.water_distance}
          onChange={(e) =>
            setForm({ ...form, water_distance: e.target.value })
          }
        />

        <label>Distance to Forest (m)</label>
        <input
          type="number"
          value={form.distance_to_forest}
          onChange={(e) =>
            setForm({ ...form, distance_to_forest: e.target.value })
          }
        />

        <button onClick={handlePredict}>Predict Risk</button>

        {result && (
          <div
            className={
              result.message
                ? "result warning"
                : `result ${result.risk_level?.toLowerCase()}`
            }
          >
            {result.message ? (
              <p>{result.message}</p>
            ) : (
              <>
                <p>
                  <strong>Risk Score:</strong> {result.risk_score}
                </p>
                <p>
                  <strong>Risk Level:</strong> {result.risk_level}
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
