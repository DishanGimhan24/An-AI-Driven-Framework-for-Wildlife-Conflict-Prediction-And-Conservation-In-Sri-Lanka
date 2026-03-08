import { Link, NavLink, Outlet, useNavigate } from "react-router-dom";
import { Home, BarChart3, FileText, Shield, LogOut, Menu, X, Settings, Flame } from "lucide-react";
import { useState } from "react";

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();
  const isOfficer = !!localStorage.getItem("officer_token");
  const currentUser = JSON.parse(localStorage.getItem("officer_user") || "{}");
  const isAdmin = currentUser.role === "admin";

  const handleLogout = () => {
    localStorage.removeItem("officer_token");
    localStorage.removeItem("officer_user");
    navigate("/kavindu");
  };

  const navItems = [
    { to: "/kavindu", icon: Home, label: "Home" },
    { to: "/kavindu/report", icon: FileText, label: "Public Report" },
  ];

  if (isOfficer) {
    navItems.push({ to: "/kavindu/dashboard", icon: BarChart3, label: "Prediction Dashboard" });
    navItems.push({ to: "/kavindu/hotspots", icon: Flame, label: "Hotspot Ranking" });
    navItems.push({ to: "/kavindu/officer/dashboard", icon: Shield, label: "Officer Portal" });
    if (isAdmin) {
      navItems.push({ to: "/kavindu/admin", icon: Settings, label: "Admin Panel" });
    }
  } else {
    navItems.push({ to: "/kavindu/officer/login", icon: Shield, label: "Officer Login" });
  }

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-header">
          <div className="logo-section">
            <div className="logo-icon">🐘</div>
            <div className="logo-text">
              <h2>Wildlife</h2>
              <span>Command Center</span>
            </div>
          </div>
          <button className="sidebar-close" onClick={() => setSidebarOpen(false)}>
            <X size={24} />
          </button>
        </div>

        <nav className="nav-menu">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
              onClick={() => setSidebarOpen(false)}
            >
              <item.icon size={20} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {isOfficer && (
          <div className="sidebar-footer">
            <button className="logout-btn" onClick={handleLogout}>
              <LogOut size={20} />
              <span>Logout</span>
            </button>
          </div>
        )}

        <div className="sidebar-branding">
          <p>Sri Lanka Wildlife Conservation</p>
          <p className="version">v2.0 • 2026</p>
        </div>
      </aside>

      {/* Main Content */}
      <div className="main-content">
        <header className="top-bar">
          <button className="menu-toggle" onClick={() => setSidebarOpen(true)}>
            <Menu size={24} />
          </button>
          <div className="top-bar-title">Wildlife Offence Prediction System</div>
          <div className="top-bar-status">
            <div className="status-indicator"></div>
            <span>System Active</span>
          </div>
        </header>

        <main className="content-area">
          <Outlet />
        </main>
      </div>

      {/* Mobile overlay */}
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)}></div>}
    </div>
  );
}
