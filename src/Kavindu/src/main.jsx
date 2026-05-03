import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Home from "./pages/Home.jsx";
import PredictionDashboard from "./pages/PredictionDashboard.jsx";
import HotspotRanking from "./pages/HotspotRanking.jsx";
import Report from "./pages/Report.jsx";
import OfficerLogin from "./pages/OfficerLogin.jsx";
import OfficerDashboard from "./pages/OfficerDashboard.jsx";
import AdminPanel from "./pages/AdminPanel.jsx";
import ProtectedRoute from "./components/ProtectedRoute.jsx";
import "./App.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route
            path="dashboard"
            element={
              <ProtectedRoute>
                <PredictionDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="hotspots"
            element={
              <ProtectedRoute>
                <HotspotRanking />
              </ProtectedRoute>
            }
          />
          <Route path="report" element={<Report />} />
          <Route path="officer/login" element={<OfficerLogin />} />
          <Route
            path="officer/dashboard"
            element={
              <ProtectedRoute>
                <OfficerDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="admin"
            element={
              <ProtectedRoute requiredRole="admin">
                <AdminPanel />
              </ProtectedRoute>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
