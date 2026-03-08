import { useState } from 'react';
import { getTodayDate } from '../../utils/helpers';
import Button from '../common/Button';
import DatePicker from '../common/DatePicker';
import CitySelector from './CitySelector';
import DistrictSelector from './DistrictSelector';

export default function PredictionForm({ onSubmit, loading }) {
  const [district, setDistrict] = useState('');
  const [city, setCity] = useState('');
  const [date, setDate] = useState(getTodayDate());
  const [errors, setErrors] = useState({});

  const validate = () => {
    const newErrors = {};
    if (!district) newErrors.district = 'Please select a district';
    if (!city) newErrors.city = 'Please select a city';
    if (!date) newErrors.date = 'Please select a date';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validate()) onSubmit({ district, city, date });
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <DistrictSelector
        value={district}
        onChange={(e) => { setDistrict(e.target.value); setCity(''); }}
        label="Select District"
        required
      />
      {errors.district && <p style={{ fontSize: '13px', color: '#f87171', marginTop: '-12px' }}>{errors.district}</p>}

      <CitySelector
        district={district}
        value={city}
        onChange={(e) => setCity(e.target.value)}
        label="Select City/Division"
      />
      {errors.city && <p style={{ fontSize: '13px', color: '#f87171', marginTop: '-12px' }}>{errors.city}</p>}

      <DatePicker
        label="Select Date"
        value={date}
        onChange={(e) => setDate(e.target.value)}
        min={getTodayDate()}
        required
        error={errors.date}
      />

      <Button type="submit" variant="primary" className="w-full" loading={loading} disabled={loading}>
        Predict Risk
      </Button>
    </form>
  );
}