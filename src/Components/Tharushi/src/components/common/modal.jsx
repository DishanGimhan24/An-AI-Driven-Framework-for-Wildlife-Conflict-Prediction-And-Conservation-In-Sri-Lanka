import { X } from 'lucide-react';

const SIZES = { sm: '440px', md: '520px', lg: '760px', xl: '960px' };

export default function Modal({ isOpen, onClose, title, children, size = 'md' }) {
  if (!isOpen) return null;

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 50, overflowY: 'auto' }}>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)' }}
      />

      {/* Modal */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', padding: '16px' }}>
        <div style={{
          position: 'relative',
          background: 'rgba(17,24,39,0.95)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255,255,255,0.18)',
          borderRadius: '20px',
          boxShadow: '0 25px 50px rgba(0,0,0,0.5)',
          maxWidth: SIZES[size] || SIZES.md,
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
        }}>
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '18px 24px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
            <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#d1d5db', margin: 0 }}>{title}</h3>
            <button
              onClick={onClose}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', padding: '4px', borderRadius: '6px' }}
              onMouseEnter={e => e.currentTarget.style.color = '#d1d5db'}
              onMouseLeave={e => e.currentTarget.style.color = '#6b7280'}
            >
              <X size={20} />
            </button>
          </div>

          {/* Content */}
          <div style={{ padding: '24px' }}>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}