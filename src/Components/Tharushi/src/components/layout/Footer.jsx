export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer style={{
      background: 'var(--glass-bg)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      borderTop: '1px solid var(--glass-border)',
      padding: '24px 0',
      marginTop: 'auto',
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 24px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '16px' }}>
          <div>
            <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: '4px' }}>
              © {currentYear} ELESAFE — Wildlife Conflict Prediction System
            </p>
            <p style={{ fontSize: '11px', color: '#6b7280', margin: 0 }}>Protecting Communities &amp; Wildlife in Sri Lanka</p>
          </div>
          <div style={{ display: 'flex', gap: '24px' }}>
            {['About', 'Help', 'Contact'].map((link) => (
              <a
                key={link}
                href="#"
                style={{
                  fontSize: '13px',
                  color: '#9ca3af',
                  textDecoration: 'none',
                  transition: 'color 0.2s',
                }}
                onMouseEnter={(e) => (e.target.style.color = 'var(--emerald-400)')}
                onMouseLeave={(e) => (e.target.style.color = '#9ca3af')}
              >
                {link}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}