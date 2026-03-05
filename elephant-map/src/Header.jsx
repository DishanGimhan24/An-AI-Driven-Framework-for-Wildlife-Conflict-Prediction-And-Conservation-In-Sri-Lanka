import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { path: "/",                    label: "📊 Dashboard",         end: true },
  { path: "/map",                 label: "🗺️ Map"                          },
  { path: "/hotspots",            label: "🐘 Hotspots"                     },
  { path: "/corridors",           label: "🛤️ Corridors"                    },
  { path: "/road-crossings",      label: "🚗 Road Crossings"               },
  { path: "/road-crossings-map",  label: "📍 Crossings Map"                },
  { path: "/predict",             label: "🔮 Risk Prediction", cta: true  },
];

export default function Header() {
  return (
    <header style={{
      position: "fixed",
      top: 0, left: 0, right: 0,
      height: "60px",
      background: "linear-gradient(90deg, #0d1b5e 0%, #1a237e 60%, #283593 100%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0 20px",
      zIndex: 3000,
      boxShadow: "0 2px 12px rgba(0,0,0,0.4)"
    }}>
      {/* Brand */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px", flexShrink: 0 }}>
        <span style={{ fontSize: "30px", lineHeight: 1 }}>🐘</span>
        <div>
          <div style={{ color: "white", fontSize: "15px", fontWeight: "700", lineHeight: 1.2, letterSpacing: "0.3px" }}>
            Wildlife Conflict Prediction
          </div>
          <div style={{ color: "rgba(255,255,255,0.6)", fontSize: "10px", lineHeight: 1.3, letterSpacing: "0.5px" }}>
            Sri Lanka · AI-Driven Conservation Framework
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav style={{ display: "flex", gap: "2px", alignItems: "center" }}>
        {NAV_ITEMS.map(item => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.end}
            style={({ isActive }) => ({
              color: "white",
              textDecoration: "none",
              padding: "7px 13px",
              borderRadius: "6px",
              fontSize: "12.5px",
              fontWeight: isActive ? "700" : "500",
              backgroundColor: isActive
                ? "rgba(255,255,255,0.22)"
                : item.cta
                  ? "rgba(255,255,255,0.12)"
                  : "transparent",
              border: item.cta ? "1px solid rgba(255,255,255,0.35)" : "1px solid transparent",
              transition: "background 0.15s, border-color 0.15s",
              whiteSpace: "nowrap",
              letterSpacing: "0.2px"
            })}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
