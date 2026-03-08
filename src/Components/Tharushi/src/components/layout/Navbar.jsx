import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { logout } from '../../api/authAPI';
import { useApp } from '../../context/AppContext';
import { Menu, X } from 'lucide-react';

export default function Navbar() {
  const { user, logout: logoutContext, isLoggedIn } = useApp();
  const navigate = useNavigate();
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    logoutContext();
    navigate('/tharushi/login');
  };

  if (location.pathname === '/tharushi/login') {
    return null;
  }

  const navLinks = [
    { to: '/tharushi/dashboard', label: 'Dashboard' },
    { to: '/tharushi/predict', label: 'Predict Risk' },
    { to: '/tharushi/map-calendar', label: 'Map & Calendar' },
    { to: '/tharushi/historical', label: 'Historical Data' },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="glass-navbar">
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '64px' }}>
          {/* Logo */}
          <Link
            to="/tharushi/dashboard"
            style={{ display: 'flex', alignItems: 'center', gap: '12px', textDecoration: 'none' }}
          >
            <div style={{
              background: 'linear-gradient(135deg, var(--emerald-600), var(--emerald-800))',
              borderRadius: '50%',
              width: '44px',
              height: '44px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '22px',
              boxShadow: '0 0 12px rgba(16,185,129,0.4)',
            }}>
              🐘
            </div>
            <div>
              <h1 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--emerald-400)', margin: 0, lineHeight: 1.2 }}>ELESAFE</h1>
              <p style={{ fontSize: '11px', color: '#9ca3af', margin: 0, textTransform: 'uppercase', letterSpacing: '1px' }}>Conflict Prediction</p>
            </div>
          </Link>

          {/* Desktop Nav */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }} className="hidden md:flex">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                style={{
                  padding: '8px 16px',
                  borderRadius: '10px',
                  fontSize: '14px',
                  fontWeight: 500,
                  textDecoration: 'none',
                  transition: 'all 0.2s',
                  color: isActive(link.to) ? 'var(--emerald-400)' : '#d1d5db',
                  background: isActive(link.to) ? 'rgba(16,185,129,0.15)' : 'transparent',
                }}
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* Right side */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {isLoggedIn && (
              <>
                <div style={{ textAlign: 'right', display: 'none' }} className="hidden md:block">
                  <p style={{ fontSize: '13px', fontWeight: 600, color: '#f3f4f6', margin: 0 }}>{user?.username || 'User'}</p>
                  <p style={{ fontSize: '11px', color: '#9ca3af', margin: 0 }}>Wildlife Officer</p>
                </div>
                <button
                  onClick={handleLogout}
                  style={{
                    padding: '8px 16px',
                    background: 'rgba(239,68,68,0.1)',
                    border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: '10px',
                    color: '#fca5a5',
                    fontSize: '13px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                >
                  Logout
                </button>
              </>
            )}
            {/* Mobile menu button */}
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              style={{ background: 'none', border: 'none', color: '#d1d5db', cursor: 'pointer', padding: '8px', display: 'none' }}
              className="block md:hidden"
            >
              {menuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {menuOpen && (
          <div style={{ padding: '12px 0 16px', borderTop: '1px solid rgba(255,255,255,0.1)' }} className="md:hidden">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                onClick={() => setMenuOpen(false)}
                style={{
                  display: 'block',
                  padding: '10px 8px',
                  fontSize: '14px',
                  fontWeight: 500,
                  textDecoration: 'none',
                  color: isActive(link.to) ? 'var(--emerald-400)' : '#d1d5db',
                }}
              >
                {link.label}
              </Link>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
}