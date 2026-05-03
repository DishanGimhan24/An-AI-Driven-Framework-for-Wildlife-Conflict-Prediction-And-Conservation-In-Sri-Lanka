import { Loader2 } from 'lucide-react';

export default function Loading({ size = 'md', text = 'Loading...' }) {
  const pxNum = { sm: 24, md: 40, lg: 64 }[size] || 40;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '32px 0' }}>
      <Loader2
        size={pxNum}
        style={{ animation: 'spin 1s linear infinite', color: 'var(--emerald-500)' }}
      />
      {text && (
        <p style={{ marginTop: '12px', fontSize: '14px', color: '#9ca3af' }}>{text}</p>
      )}
    </div>
  );
}