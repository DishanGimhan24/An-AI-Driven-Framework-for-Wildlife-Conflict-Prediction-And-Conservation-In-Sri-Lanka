import { useEffect, useState } from 'react';
import { getCitiesByDistrict } from '../../api/cityAPI';

export default function CitySelector({ district, value, onChange, label = 'Select City/Division', disabled = false }) {
  const [cities, setCities] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (district) {
      loadCities();
    } else {
      setCities([]);
    }
  }, [district]);

  const loadCities = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await getCitiesByDistrict(district);
      if (response.status === 'success') {
        setCities(response.data.cities);
      }
    } catch (err) {
      setError('Failed to load cities');
      setCities([]);
    } finally {
      setLoading(false);
    }
  };

  const isDisabled = disabled || loading || !district;

  return (
    <div style={{ marginBottom: '16px' }}>
      <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#d1d5db', marginBottom: '8px' }}>
        {label}
      </label>
      <select
        value={value}
        onChange={onChange}
        disabled={isDisabled}
        className="glass-input"
        style={{ opacity: isDisabled ? 0.6 : 1, cursor: isDisabled ? 'not-allowed' : 'auto' }}
      >
        <option value="" style={{ background: '#111827' }}>
          {loading ? 'Loading cities...' : !district ? 'Select district first' : cities.length === 0 ? 'No cities available' : 'Select a city'}
        </option>
        {cities.map((city) => (
          <option key={city.gid} value={city.name} style={{ background: '#111827' }}>
            {city.name}
          </option>
        ))}
      </select>
      {error && <p style={{ marginTop: '6px', fontSize: '13px', color: '#f87171' }}>{error}</p>}
      {!loading && !error && cities.length > 0 && district && (
        <p style={{ marginTop: '4px', fontSize: '12px', color: '#6b7280' }}>{cities.length} cities available</p>
      )}
    </div>
  );
}