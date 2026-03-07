import { BrowserRouter as Router, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import './App.css';
import './Components/Tharushi/src/index.css';

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

// ── Tharushi (ELESAFE) pages ──────────────────
import { AppProvider, useApp } from './Components/Tharushi/src/context/AppContext';
import TharushiLogin        from './Components/Tharushi/src/pages/Login';          // /tharushi/login
import TharushiDashboard    from './Components/Tharushi/src/pages/Dashboard';      // /tharushi/dashboard
import TharushiPredict      from './Components/Tharushi/src/pages/RiskPrediction'; // /tharushi/predict
import TharushiMapCalendar  from './Components/Tharushi/src/pages/MapCalendar';    // /tharushi/map-calendar
import TharushiHistorical   from './Components/Tharushi/src/pages/Historical';     // /tharushi/historical

function TharushiLayout() {
  return <AppProvider><Outlet /></AppProvider>;
}

function TharushiProtected({ children }) {
  const { isLoggedIn } = useApp();
  return isLoggedIn ? children : <Navigate to="/tharushi/login" replace />;
}

function TharushiPublic({ children }) {
  const { isLoggedIn } = useApp();
  return isLoggedIn ? <Navigate to="/tharushi/dashboard" replace /> : children;
}

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

          {/* ── Tharushi (ELESAFE) routes ── */}
          <Route path="/tharushi" element={<TharushiLayout />}>
            <Route index                element={<Navigate to="dashboard" replace />} />
            <Route path="login"         element={<TharushiPublic><TharushiLogin /></TharushiPublic>} />
            <Route path="dashboard"     element={<TharushiProtected><TharushiDashboard /></TharushiProtected>} />
            <Route path="predict"       element={<TharushiProtected><TharushiPredict /></TharushiProtected>} />
            <Route path="map-calendar"  element={<TharushiProtected><TharushiMapCalendar /></TharushiProtected>} />
            <Route path="historical"    element={<TharushiProtected><TharushiHistorical /></TharushiProtected>} />
          </Route>
        </Routes>
      </div>
    </Router>
  );
}

export default App;
