import { Link, useLocation } from 'react-router-dom';
import { Home, BarChart3, Map, Clock } from 'lucide-react';

export default function Sidebar() {
  const location = useLocation();
  const isActive = (path) => location.pathname === path;

  const menuItems = [
    { path: '/tharushi/dashboard', label: 'Dashboard', icon: 'home' },
    { path: '/tharushi/predict', label: 'Predict Risk', icon: 'chart' },
    { path: '/tharushi/map-calendar', label: 'Map & Calendar', icon: 'map' },
    { path: '/tharushi/historical', label: 'Historical Data', icon: 'history' },
  ];

  const icons = {
    home: <Home size={20} />,
    chart: <BarChart3 size={20} />,
    map: <Map size={20} />,
    history: <Clock size={20} />,
  };

  return (
    <aside style={{
      width: '240px',
      background: 'var(--glass-bg)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      borderRight: '1px solid var(--glass-border)',
      position: 'sticky',
      top: '64px',
      height: 'calc(100vh - 64px)',
      overflowY: 'auto',
    }} className="hidden lg:block">
      <nav style={{ padding: '16px 12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {menuItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '12px 16px',
              borderRadius: '12px',
              color: isActive(item.path) ? 'var(--emerald-400)' : '#d1d5db',
              background: isActive(item.path) ? 'rgba(16,185,129,0.15)' : 'transparent',
              textDecoration: 'none',
              fontWeight: 500,
              fontSize: '14px',
              transition: 'all 0.2s',
              borderLeft: isActive(item.path) ? '3px solid var(--emerald-500)' : '3px solid transparent',
            }}
          >
            {icons[item.icon]}
            <span>{item.label}</span>
          </Link>
        ))}
      </nav>
    </aside>
  );
}