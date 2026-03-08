import { useState } from "react";
import "./HimashiSidebar.css";
import { NavLink } from "react-router-dom";
import {
  BarChart3,
  Map,
  Brain,
  TrendingUp,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Flame,
} from "lucide-react";

const NAV_SECTIONS = [
  {
    label: "Analytics",
    items: [
      { to: "/risk-dashboard", icon: BarChart3, label: "Overview" },
      { to: "/top-districts", icon: TrendingUp, label: "Top Districts" },
    ],
  },
  {
    label: "Geospatial",
    items: [
      { to: "/risk-map", icon: Map, label: "Risk Map" },
      { to: "/hotspots", icon: Flame, label: "Hotspots" },
    ],
  },
  {
    label: "Tools",
    items: [
      { to: "/risk-prediction", icon: Brain, label: "Prediction" },
    ],
  },
];

export default function HimashiSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside className={`sidebar${collapsed ? " sidebar--collapsed" : ""}`}>
      {/* ── Header ── */}
      <div className="sidebar__header">
        <div className="sidebar__logo-icon">🐘</div>
        <div className="sidebar__logo-text">
          <h2>Wildlife Guard</h2>
          <span>Command Center</span>
        </div>
        <button
          className="sidebar__toggle"
          onClick={() => setCollapsed(!collapsed)}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {/* ── User ── */}
      <div className="sidebar__user">
        <img
          className="sidebar__avatar"
          src="https://i.pravatar.cc/80?img=12"
          alt="Officer"
        />
        <div className="sidebar__user-info">
          <div className="sidebar__name">Wildlife Officer</div>
          <div className="sidebar__status">
            <span className="sidebar__dot" />
            <span className="sidebar__status-text">Online</span>
          </div>
        </div>
      </div>

      {/* ── Navigation ── */}
      <nav className="sidebar__nav">
        {NAV_SECTIONS.map((section) => (
          <div key={section.label} className="sidebar__section">
            <div className="sidebar__section-label">{section.label}</div>
            {section.items.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  "sidebar__link" + (isActive ? " active" : "")
                }
                title={label}
              >
                <Icon size={18} className="sidebar__link-icon" />
                <span className="sidebar__link-text">{label}</span>
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      {/* ── Footer ── */}
      <div className="sidebar__footer">
        <div className="sidebar__version">
          <AlertTriangle size={12} />
          <span className="sidebar__version-text">AVC System v2.0</span>
        </div>
      </div>
    </aside>
  );
}
