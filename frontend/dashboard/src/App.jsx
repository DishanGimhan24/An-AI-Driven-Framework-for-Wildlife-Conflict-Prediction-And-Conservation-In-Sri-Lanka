import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import RiskMap from "./RiskMap.jsx";
import Prediction from "./pages/Prediction";
import Dashboard from "./pages/Dashboard";

function App() {
  return (
    <div style={{ width: "100%", minHeight: "100vh" }}>
      <Navbar />

      <Routes>
        {/* Home */}
        <Route path="/" element={<Home />} />

        {/* Dashboard */}
        <Route path="/dashboard" element={<Dashboard />} />

        {/* Prediction inside dashboard layout */}
        <Route path="/dashboard/prediction" element={<Dashboard />}>
          <Route index element={<Prediction />} />
        </Route>

        {/* Full screen map */}
        <Route path="/map" element={<RiskMap />} />
      </Routes>
    </div>
  );
}

export default App;