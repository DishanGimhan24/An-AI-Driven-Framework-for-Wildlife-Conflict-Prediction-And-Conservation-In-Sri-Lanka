export default function Card({ children, title, className = '', padding = '', shadow = '' }) {
  return (
    <div
      className={`glass-card ${className}`}
      style={padding ? { padding } : undefined}
    >
      {title && (
        <h3
          style={{
            fontSize: '18px',
            fontWeight: 700,
            color: 'var(--emerald-400)',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}