import { Navigate, Route, Routes } from 'react-router-dom';
import './index.css';
import { AppProvider, useApp } from './context/AppContext';
import Dashboard from './pages/Dashboard';
import Historical from './pages/Historical';
import Login from './pages/Login';
import MapCalendar from './pages/MapCalendar';
import RiskPrediction from './pages/RiskPrediction';

// Protected Route Component
function ProtectedRoute({ children }) {
  const { isLoggedIn } = useApp();
  
  if (!isLoggedIn) {
    return <Navigate to="/tharushi/login" replace />;
  }
  
  return children;
}

// Public Route Component (redirect to dashboard if already logged in)
function PublicRoute({ children }) {
  const { isLoggedIn } = useApp();
  
  if (isLoggedIn) {
    return <Navigate to="/tharushi/dashboard" replace />;
  }
  
  return children;
}

function AppRoutes() {
  return (
    <Routes>
      {/* Public Routes */}
      <Route
        path="login"
        element={
          <PublicRoute>
            <Login />
          </PublicRoute>
        }
      />

      {/* Protected Routes */}
      <Route
        path="dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />

      <Route
        path="predict"
        element={
          <ProtectedRoute>
            <RiskPrediction />
          </ProtectedRoute>
        }
      />

      <Route
        path="map-calendar"
        element={
          <ProtectedRoute>
            <MapCalendar />
          </ProtectedRoute>
        }
      />

      <Route
        path="historical"
        element={
          <ProtectedRoute>
            <Historical />
          </ProtectedRoute>
        }
      />

      {/* Default Route */}
      <Route path="" element={<Navigate to="dashboard" replace />} />

      {/* 404 Not Found */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

// 404 Page Component
function NotFound() {
  const { isLoggedIn } = useApp();

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px', position: 'relative', zIndex: 1 }}>
      <div style={{ textAlign: 'center' }}>
        <h1 className="page-title-gradient" style={{ fontSize: '5rem', fontWeight: 800, marginBottom: '16px' }}>404</h1>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#d1d5db', marginBottom: '16px' }}>
          Page Not Found
        </h2>
        <p style={{ color: '#9ca3af', marginBottom: '32px' }}>
          The page you&apos;re looking for doesn&apos;t exist.
        </p>
        <a
          href={isLoggedIn ? "/tharushi/dashboard" : "/tharushi/login"}
          className="btn-emerald"
          style={{ display: 'inline-block', textDecoration: 'none', padding: '12px 28px', borderRadius: '10px' }}
        >
          {isLoggedIn ? "Go to Dashboard" : "Go to Login"}
        </a>
      </div>
    </div>
  );
}

function App() {
  return (
    <AppProvider>
      <AppRoutes />
    </AppProvider>
  );
}

export default App;