import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

// ── Core layout ──────────────────────────────
import Header               from './Header';

// ── Elephant-map pages ───────────────────────
import DashboardPage        from './DashboardPage';        // /
import ElephantMap          from './ElephantMap';           // /map
import HotspotsPage         from './Himashi/HotspotsPage';          // /hotspots
import CorridorsPage        from './CorridorsPage';          // /corridors
import RoadCrossingsPage    from './RoadCrossingsPage';      // /road-crossings
import RoadCrossingsMapPage from './RoadCrossingsMapPage';   // /road-crossings-map
import PredictPage          from './PredictPage';            // /predict

// ── Himashi pages (merged from second branch) ─
import HimashiHome          from './Himashi/HimashiHome';            // /avc-home
import HimashiDashboard     from './Himashi/HimashiDashboard';       // /risk-dashboard
import HimashiRiskMap       from './Himashi/HimashiRiskMap';         // /risk-map
import HimashiPrediction    from './Himashi/HimashiPrediction';      // /risk-prediction
import HimashiDistrictDetails from './Himashi/HimashiDistrictDetails'; // /district/:district
import HimashiTopDistricts    from './Himashi/HimashiTopDistricts';    // /top-districts
import HimashiRiskSummary     from './Himashi/HimashiRiskSummary';     // /risk-summary
import HimashiRiskLocations   from './Himashi/HimashiRiskLocations';   // /risk-locations

function App() {
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Header />
      <div style={{ marginTop: '60px' }}>
        <Routes>
          {/* ── Elephant-map routes ── */}
          <Route path="/"                   element={<DashboardPage />} />
          <Route path="/map"                element={<ElephantMap />} />
          <Route path="/hotspots"           element={<HotspotsPage />} />
          <Route path="/corridors"          element={<CorridorsPage />} />
          <Route path="/road-crossings"     element={<RoadCrossingsPage />} />
          <Route path="/road-crossings-map" element={<RoadCrossingsMapPage />} />
          <Route path="/predict"            element={<PredictPage />} />

          {/* ── Himashi (AVC) routes ── */}
          <Route path="/avc-home"           element={<HimashiHome />} />
          <Route path="/risk-dashboard"     element={<HimashiDashboard />} />
          <Route path="/risk-map"           element={<HimashiRiskMap />} />
          <Route path="/risk-prediction"    element={<HimashiPrediction />} />
          <Route path="/district/:district" element={<HimashiDistrictDetails />} />
          <Route path="/top-districts"     element={<HimashiTopDistricts />} />
          <Route path="/risk-summary"       element={<HimashiRiskSummary />} />
          <Route path="/risk-locations"     element={<HimashiRiskLocations />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
