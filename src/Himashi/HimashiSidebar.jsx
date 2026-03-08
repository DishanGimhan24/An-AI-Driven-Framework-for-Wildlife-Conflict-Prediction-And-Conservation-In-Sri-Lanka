import "./HimashiSidebar.css";
import { NavLink } from "react-router-dom";
import { BarChart3, Map, Brain } from "lucide-react";

export default function HimashiSidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar__header">
        <div className="sidebar__logo-icon">🐘</div>
        <div className="sidebar__logo-text">
          <h2>Wildlife Guard</h2>
          <span>Command Center</span>
        </div>
      </div>

      <div className="sidebar__user">
        <img
          className="sidebar__avatar"
          src="https://i.pravatar.cc/80?img=12"
          alt="Officer"
        />
        <div>
          <div className="sidebar__name">Wildlife Officer</div>
          <div className="sidebar__status">
            <span className="sidebar__dot" /> Online
          </div>
        </div>
      </div>

      <nav className="sidebar__nav">
        <NavLink
          to="/risk-dashboard"
          className={({ isActive }) =>
            "sidebar__link" + (isActive ? " active" : "")
          }
        >
          <BarChart3 size={18} />
          Overview
        </NavLink>

        <NavLink
          to="/risk-map"
          className={({ isActive }) =>
            "sidebar__link" + (isActive ? " active" : "")
          }
        >
          <Map size={18} />
          Risk Map
        </NavLink>

        <NavLink
          to="/risk-prediction"
          className={({ isActive }) =>
            "sidebar__link" + (isActive ? " active" : "")
          }
        >
          <Brain size={18} />
          Prediction
        </NavLink>
      </nav>
    </aside>
  );
}
