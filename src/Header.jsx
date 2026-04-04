import { NavLink } from 'react-router-dom';
import './Header.css';

export default function Header() {
  return (
    <header className="app-header">
      <span className="app-header__brand">
        <span className="app-header__brand-icon">🐘</span>
        Wildlife Conflict AI
      </span>
      <div className="app-header__divider" />
      <nav className="app-header__nav">
        <NavLink to="/"                className={({ isActive }) => isActive ? 'active' : ''}>Dashboard</NavLink>
        <NavLink to="/map"             className={({ isActive }) => isActive ? 'active' : ''}>Map</NavLink>
        <NavLink to="/hotspots"        className={({ isActive }) => isActive ? 'active' : ''}>Hotspots</NavLink>
        <NavLink to="/corridors"       className={({ isActive }) => isActive ? 'active' : ''}>Corridors</NavLink>
        <NavLink to="/road-crossings"  className={({ isActive }) => isActive ? 'active' : ''}>Road Crossings</NavLink>
        <NavLink to="/road-crossings-map" className={({ isActive }) => isActive ? 'active' : ''}>Crossings Map</NavLink>
        <NavLink to="/predict"         className={({ isActive }) => isActive ? 'active' : ''}>Predict</NavLink>
        <NavLink to="/avc-home"        className={({ isActive }) => isActive ? 'active' : ''}>AVC Home</NavLink>
        <NavLink to="/risk-dashboard"  className={({ isActive }) => isActive ? 'active' : ''}>Risk Dashboard</NavLink>
        <NavLink to="/risk-map"        className={({ isActive }) => isActive ? 'active' : ''}>Risk Map</NavLink>
        <NavLink to="/risk-prediction" className={({ isActive }) => isActive ? 'active' : ''}>Risk Prediction</NavLink>
      </nav>
      <div className="app-header__status">
        <div className="app-header__status-dot" />
        <span>Sri Lanka</span>
      </div>
    </header>
  );
}
