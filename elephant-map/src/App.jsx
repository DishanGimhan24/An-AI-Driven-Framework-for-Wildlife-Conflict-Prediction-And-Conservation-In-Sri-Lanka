import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

// ── Core layout ──────────────────────────────
import Header               from './Header';

// ── Elephant-map pages ───────────────────────
import DashboardPage        from './DashboardPage';        // /
import ElephantMap          from './ElephantMap';           // /map
import HotspotsPage         from './HotspotsPage';          // /hotspots
import CorridorsPage        from './CorridorsPage';          // /corridors
import RoadCrossingsPage    from './RoadCrossingsPage';      // /road-crossings
import RoadCrossingsMapPage from './RoadCrossingsMapPage';   // /road-crossings-map
import PredictPage          from './PredictPage';            // /predict

// ── Himashi pages (merged from second branch) ─
import HimashiHome          from './HimashiHome';            // /avc-home
import HimashiDashboard     from './HimashiDashboard';       // /risk-dashboard
import HimashiRiskMap       from './HimashiRiskMap';         // /risk-map
import HimashiPrediction    from './HimashiPrediction';      // /risk-prediction

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
        </Routes>
      </div>
    </Router>
  );
}

export default App;
