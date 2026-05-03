import { useState } from 'react';
import { Search } from 'lucide-react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import Button from '../components/common/Button';
import Card from '../components/common/Card';
import ErrorMessage from '../components/common/ErrorMessage';
import Loading from '../components/common/Loading';
import CityRiskMap from '../components/features/CityRiskMap';
import PredictionForm from '../components/features/PredictionForm';
import RiskBadge from '../components/features/RiskBadge';
import Footer from '../components/layout/Footer';
import Navbar from '../components/layout/Navbar';
import { useForecast } from '../hooks/useForecast';
import { usePrediction } from '../hooks/usePrediction';
import { formatDateDisplay } from '../utils/helpers';

export default function RiskPrediction() {
  const [forecastDays, setForecastDays] = useState(7);
  const { prediction, loading: predictionLoading, error: predictionError, predict, clearPrediction } = usePrediction();
  const { forecast, loading: forecastLoading, getForecast } = useForecast();

  const handlePredict = async (formData) => {
    const result = await predict(formData.district, formData.city, formData.date);
    if (result && result.coordinates) {
      await getForecast(result.coordinates.lat, result.coordinates.lng, forecastDays, formData.date);
    }
  };

  const handleForecastDaysChange = async (days) => {
    setForecastDays(days);
    if (prediction && prediction.coordinates) {
      await getForecast(prediction.coordinates.lat, prediction.coordinates.lng, days, prediction.date);
    }
  };

  const getChartData = () => {
    if (!forecast || !forecast.forecast) return [];
    return forecast.forecast.map(day => ({
      date: formatDateDisplay(day.date),
      risk: Math.round(day.risk_score * 100),
      day: day.day,
    }));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1 }}>
      <Navbar />

      <main style={{ flex: 1, maxWidth: '1400px', margin: '0 auto', padding: '32px 24px', width: '100%' }}>
        <div style={{ marginBottom: '32px' }}>
          <h1 className="page-title-gradient" style={{ fontSize: '2rem', marginBottom: '6px' }}>Risk Prediction</h1>
          <p style={{ color: '#9ca3af', fontSize: '15px' }}>Predict wildlife conflict risk for any city and date</p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px', alignItems: 'start' }}>
          {/* Input Form */}
          <div>
            <Card title="Enter Location Details">
              <PredictionForm onSubmit={handlePredict} loading={predictionLoading} />
              {prediction && (
                <Button onClick={clearPrediction} variant="outline" style={{ width: '100%', marginTop: '16px' }}>
                  Clear &amp; Predict Again
                </Button>
              )}
            </Card>
            {predictionError && <div style={{ marginTop: '16px' }}><ErrorMessage message={predictionError} /></div>}
          </div>

          {/* Results */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', gridColumn: 'span 2' }}>
            {predictionLoading ? (
              <Card><Loading text="Analyzing risk factors..." /></Card>
            ) : prediction ? (
              <>
                {/* Risk Result */}
                <Card title="Prediction Result">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
                      <div>
                        <h3 style={{ fontSize: '20px', fontWeight: 700, color: '#f3f4f6', marginBottom: '4px' }}>
                          {prediction.city}, {prediction.district}
                        </h3>
                        <p style={{ fontSize: '13px', color: '#9ca3af' }}>Date: {formatDateDisplay(prediction.date)}</p>
                      </div>
                      <RiskBadge level={prediction.risk_level} size="large" />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', padding: '16px', background: 'rgba(255,255,255,0.04)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)' }}>
                      <div>
                        <p style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '4px' }}>Risk Score</p>
                        <p style={{ fontSize: '2rem', fontWeight: 800, color: 'white' }}>{Math.round(prediction.risk_score * 100)}%</p>
                      </div>
                      <div>
                        <p style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '4px' }}>Confidence</p>
                        <p style={{ fontSize: '2rem', fontWeight: 800, color: 'white' }}>{Math.round(prediction.confidence * 100)}%</p>
                      </div>
                    </div>

                    {prediction.contributing_factors && (
                      <div>
                        <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--emerald-400)', marginBottom: '10px' }}>Contributing Factors</h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                          {Object.entries(prediction.contributing_factors).map(([key, value]) => (
                            <div key={key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                              <span style={{ color: '#9ca3af' }}>{key.replace(/_/g, ' ').toUpperCase()}</span>
                              <span style={{ fontWeight: 600, color: '#e5e7eb' }}>{value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </Card>

                {/* City Map */}
                <Card title="Location Map">
                  <CityRiskMap
                    district={prediction.district}
                    city={prediction.city}
                    riskLevel={prediction.risk_level}
                    riskScore={prediction.risk_score}
                  />
                </Card>

                {/* Forecast */}
                <Card title="Risk Forecast">
                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#d1d5db', marginBottom: '10px' }}>Forecast Days</label>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      {[7, 14, 30].map(days => (
                        <button
                          key={days}
                          onClick={() => handleForecastDaysChange(days)}
                          style={{
                            padding: '8px 18px',
                            borderRadius: '10px',
                            border: 'none',
                            fontWeight: 600,
                            fontSize: '13px',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            background: forecastDays === days
                              ? 'linear-gradient(135deg, var(--emerald-600), var(--emerald-700))'
                              : 'rgba(255,255,255,0.08)',
                            color: forecastDays === days ? 'white' : '#9ca3af',
                          }}
                        >
                          {days} Days
                        </button>
                      ))}
                    </div>
                  </div>

                  {forecastLoading ? (
                    <Loading text="Generating forecast..." />
                  ) : forecast && forecast.forecast ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={getChartData()}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} angle={-45} textAnchor="end" height={80} />
                        <YAxis label={{ value: 'Risk %', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} tick={{ fill: '#9ca3af' }} />
                        <Tooltip
                          contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.18)', borderRadius: '10px', color: '#e5e7eb' }}
                        />
                        <Legend wrapperStyle={{ color: '#9ca3af' }} />
                        <Line type="monotone" dataKey="risk" stroke="var(--emerald-500)" strokeWidth={2} dot={{ fill: 'var(--emerald-500)' }} name="Risk Score %" />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : null}
                </Card>
              </>
            ) : (
              <Card>
                <div style={{ padding: '48px 0', textAlign: 'center' }}>
                  <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'center', color: '#4b5563' }}>
                    <Search size={48} />
                  </div>
                  <p style={{ color: '#6b7280', fontSize: '15px' }}>Select a city and date to predict risk</p>
                </div>
              </Card>
            )}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

