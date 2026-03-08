import { XCircle, X } from 'lucide-react';

export default function ErrorMessage({ message, onClose }) {
  if (!message) return null;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: '12px',
      padding: '14px 16px',
      background: 'rgba(239,68,68,0.1)',
      border: '1px solid rgba(239,68,68,0.3)',
      borderRadius: '12px',
      color: '#fca5a5',
      marginBottom: '16px',
      fontSize: '14px',
    }}>
      <XCircle size={20} style={{ flexShrink: 0, marginTop: '1px', color: '#f87171' }} />
      <div style={{ flex: 1 }}>
        <p style={{ fontWeight: 500, margin: 0 }}>{message}</p>
      </div>
      {onClose && (
        <button
          onClick={onClose}
          style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer', padding: '2px' }}
        >
          <X size={20} />
        </button>
      )}
    </div>
  );
}