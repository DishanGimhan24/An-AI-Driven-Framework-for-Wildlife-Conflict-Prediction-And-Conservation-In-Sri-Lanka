import "./Sidebar.css";
import { NavLink } from "react-router-dom";

export default function Sidebar() {
  return (
    <aside className="sidebar">
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
          to="/dashboard"
          className={({ isActive }) =>
            "sidebar__link" + (isActive ? " active" : "")
          }
        >
          Overview
        </NavLink>

        {/* ✅ මෙන්න Risk Map click කරාම /map යනවා */}
        <NavLink
          to="/map"
          className={({ isActive }) =>
            "sidebar__link" + (isActive ? " active" : "")
          }
        >
          Risk Map
        </NavLink>

        <NavLink
          to="/prediction"
          className={({ isActive }) =>
            "sidebar__link" + (isActive ? " active" : "")
          }
        >
          Prediction
        </NavLink>
      </nav>
    </aside>
  );
}