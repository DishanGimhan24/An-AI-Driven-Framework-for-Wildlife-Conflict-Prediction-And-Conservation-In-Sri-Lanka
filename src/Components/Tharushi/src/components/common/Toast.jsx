import { useEffect } from 'react';
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react';

const TOAST_COLORS = {
  success: { bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)', text: '#6ee7b7', icon: '#34d399' },
  error:   { bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.3)',  text: '#fca5a5', icon: '#f87171' },
  warning: { bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)', text: '#fde68a', icon: '#fbbf24' },
  info:    { bg: 'rgba(99,102,241,0.12)', border: 'rgba(99,102,241,0.3)', text: '#c7d2fe', icon: '#818cf8' },
};

export default function Toast({ message, type = 'success', onClose, duration = 3000 }) {
  useEffect(() => {
    if (message && duration > 0) {
      const timer = setTimeout(() => { onClose(); }, duration);
      return () => clearTimeout(timer);
    }
  }, [message, duration, onClose]);

  if (!message) return null;

  const colors = TOAST_COLORS[type] || TOAST_COLORS.info;

  return (
    <div style={{ position: 'fixed', top: '20px', right: '20px', zIndex: 9999, animation: 'slideIn 0.3s ease-out' }}>
      <div style={{
        background: colors.bg,
        backdropFilter: 'blur(20px)',
        border: `1px solid ${colors.border}`,
        borderRadius: '12px',
        padding: '14px 18px',
        maxWidth: '360px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '12px',
      }}>
        <span style={{ flexShrink: 0, color: colors.icon, display: 'flex', alignItems: 'center' }}>
          {type === 'success' ? <CheckCircle size={18} /> : type === 'error' ? <XCircle size={18} /> : type === 'warning' ? <AlertTriangle size={18} /> : <Info size={18} />}
        </span>
        <p style={{ fontSize: '14px', color: colors.text, flex: 1, margin: 0, lineHeight: '1.5' }}>{message}</p>
        <button
          onClick={onClose}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', padding: '2px', flexShrink: 0, display: 'flex', alignItems: 'center' }}
          onMouseEnter={e => e.currentTarget.style.color = '#d1d5db'}
          onMouseLeave={e => e.currentTarget.style.color = '#6b7280'}
        >
          <X size={16} />
        </button>
      </div>
    </div>
  );
}
