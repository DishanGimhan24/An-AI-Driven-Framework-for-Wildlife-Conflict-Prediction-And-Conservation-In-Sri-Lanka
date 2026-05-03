import { BrowserRouter as Router, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import './App.css';
import './Components/Tharushi/src/index.css';
import './Kavindu/src/App.css';

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

// ── Tharushi (ELESAFE) pages ──────────────────
import { AppProvider, useApp } from './Components/Tharushi/src/context/AppContext';
import TharushiLogin        from './Components/Tharushi/src/pages/Login';          // /tharushi/login
import TharushiDashboard    from './Components/Tharushi/src/pages/Dashboard';      // /tharushi/dashboard
import TharushiPredict      from './Components/Tharushi/src/pages/RiskPrediction'; // /tharushi/predict
import TharushiMapCalendar  from './Components/Tharushi/src/pages/MapCalendar';    // /tharushi/map-calendar
import TharushiHistorical   from './Components/Tharushi/src/pages/Historical';     // /tharushi/historical
import TharushiEcoStress   from './Components/Tharushi/src/pages/EnvironmentalStress'; // /tharushi/eco-stress

// ── Kavindu (Wildlife Command Center) pages ───
import KavinduLayout           from './Kavindu/src/components/Layout';              // /kavindu layout
import KavinduProtectedRoute   from './Kavindu/src/components/ProtectedRoute';      // auth guard
import KavinduHome             from './Kavindu/src/pages/Home';                     // /kavindu
import KavinduPredictionDash   from './Kavindu/src/pages/PredictionDashboard';      // /kavindu/dashboard
import KavinduHotspotRanking   from './Kavindu/src/pages/HotspotRanking';           // /kavindu/hotspots
import KavinduReport           from './Kavindu/src/pages/Report';                   // /kavindu/report
import KavinduOfficerLogin     from './Kavindu/src/pages/OfficerLogin';             // /kavindu/officer/login
import KavinduOfficerDashboard from './Kavindu/src/pages/OfficerDashboard';         // /kavindu/officer/dashboard
import KavinduAdminPanel       from './Kavindu/src/pages/AdminPanel';               // /kavindu/admin

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
          <Route path="/district/:district" element={<HimashiDistrictDetails />} />
          <Route path="/top-districts"     element={<HimashiTopDistricts />} />
          <Route path="/risk-summary"       element={<HimashiRiskSummary />} />
          <Route path="/risk-locations"     element={<HimashiRiskLocations />} />

          {/* ── Tharushi (ELESAFE) routes ── */}
          <Route path="/tharushi" element={<TharushiLayout />}>
            <Route index                element={<Navigate to="dashboard" replace />} />
            <Route path="login"         element={<TharushiPublic><TharushiLogin /></TharushiPublic>} />
            <Route path="dashboard"     element={<TharushiProtected><TharushiDashboard /></TharushiProtected>} />
            <Route path="predict"       element={<TharushiProtected><TharushiPredict /></TharushiProtected>} />
            <Route path="map-calendar"  element={<TharushiProtected><TharushiMapCalendar /></TharushiProtected>} />
            <Route path="historical"    element={<TharushiProtected><TharushiHistorical /></TharushiProtected>} />
            <Route path="eco-stress"   element={<TharushiProtected><TharushiEcoStress /></TharushiProtected>} />
          </Route>

          {/* ── Kavindu (Wildlife Command Center) routes ── */}
          <Route path="/kavindu" element={<KavinduLayout />}>
            <Route index element={<KavinduHome />} />
            <Route path="dashboard"        element={<KavinduProtectedRoute><KavinduPredictionDash /></KavinduProtectedRoute>} />
            <Route path="hotspots"         element={<KavinduProtectedRoute><KavinduHotspotRanking /></KavinduProtectedRoute>} />
            <Route path="report"           element={<KavinduReport />} />
            <Route path="officer/login"    element={<KavinduOfficerLogin />} />
            <Route path="officer/dashboard" element={<KavinduProtectedRoute><KavinduOfficerDashboard /></KavinduProtectedRoute>} />
            <Route path="admin"            element={<KavinduProtectedRoute requiredRole="admin"><KavinduAdminPanel /></KavinduProtectedRoute>} />
          </Route>
        </Routes>
      </div>
    </Router>
  );
}

export default App;
