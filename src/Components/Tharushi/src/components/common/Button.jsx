import { Loader2 } from 'lucide-react';

export default function Button({
  children,
  onClick,
  type = 'button',
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  className = ''
}) {
  const variantStyle = {
    primary: {
      background: 'linear-gradient(135deg, var(--emerald-600) 0%, var(--emerald-700) 100%)',
      color: 'white',
      border: 'none',
      boxShadow: '0 4px 12px rgba(16,185,129,0.3)',
    },
    secondary: {
      background: 'rgba(255,255,255,0.08)',
      color: '#d1d5db',
      border: '1px solid rgba(255,255,255,0.18)',
    },
    danger: {
      background: 'rgba(239,68,68,0.15)',
      color: '#fca5a5',
      border: '1px solid rgba(239,68,68,0.4)',
    },
    outline: {
      background: 'transparent',
      color: 'var(--emerald-400)',
      border: '2px solid var(--emerald-600)',
    },
  };

  const sizeStyle = {
    sm: { padding: '6px 14px', fontSize: '13px' },
    md: { padding: '10px 20px', fontSize: '15px' },
    lg: { padding: '14px 28px', fontSize: '16px' },
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={`btn-emerald ${className}`}
      style={{
        ...variantStyle[variant],
        ...sizeStyle[size],
        opacity: disabled || loading ? 0.6 : 1,
        cursor: disabled || loading ? 'not-allowed' : 'pointer',
      }}
    >
      {loading ? (
        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
          Loading...
        </span>
      ) : children}
    </button>
  );
}