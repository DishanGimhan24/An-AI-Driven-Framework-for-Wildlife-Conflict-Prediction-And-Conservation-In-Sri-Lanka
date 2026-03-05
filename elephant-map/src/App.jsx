import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Header        from './Header';
import DashboardPage  from './DashboardPage';
import ElephantMap    from './ElephantMap';
import HotspotsPage   from './HotspotsPage';
import CorridorsPage  from './CorridorsPage';
import RoadCrossingsPage    from './RoadCrossingsPage';
import RoadCrossingsMapPage from './RoadCrossingsMapPage';
import PredictPage          from './PredictPage';

function App() {
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Header />
      <div style={{ marginTop: '60px' }}>
        <Routes>
          <Route path="/"               element={<DashboardPage />} />
          <Route path="/map"            element={<ElephantMap />} />
          <Route path="/hotspots"       element={<HotspotsPage />} />
          <Route path="/corridors"      element={<CorridorsPage />} />
          <Route path="/road-crossings"     element={<RoadCrossingsPage />} />
          <Route path="/road-crossings-map" element={<RoadCrossingsMapPage />} />
          <Route path="/predict"            element={<PredictPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
