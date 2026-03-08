export default function Input({
  label,
  type = 'text',
  value,
  onChange,
  placeholder = '',
  required = false,
  disabled = false,
  error = '',
  className = ''
}) {
  return (
    <div style={{ marginBottom: '16px' }} className={className}>
      {label && (
        <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#d1d5db', marginBottom: '8px' }}>
          {label}
          {required && <span style={{ color: '#f87171', marginLeft: '4px' }}>*</span>}
        </label>
      )}
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        disabled={disabled}
        className="glass-input"
        style={{
          borderColor: error ? 'rgba(239,68,68,0.6)' : undefined,
        }}
      />
      {error && (
        <p style={{ marginTop: '6px', fontSize: '13px', color: '#f87171' }}>{error}</p>
      )}
    </div>
  );
}