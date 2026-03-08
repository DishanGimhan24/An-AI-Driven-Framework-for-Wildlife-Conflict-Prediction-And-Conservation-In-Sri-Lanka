import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { RefreshCw, AlertTriangle, Info, CheckCircle, BarChart3, Map, TrendingUp, MapPin } from 'lucide-react';
import { getStats } from '../api/statsAPI';
import { checkHealth } from '../api/systemAPI';
import Card from '../components/common/Card';
import ErrorMessage from '../components/common/ErrorMessage';
import Loading from '../components/common/Loading';
import StatCard from '../components/features/StatCard';
import Footer from '../components/layout/Footer';
import Navbar from '../components/layout/Navbar';
import { formatDate } from '../utils/helpers';

export default function Dashboard() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    setError('');
    try {
      const [healthData, statsData] = await Promise.all([checkHealth(), getStats()]);
      setSystemStatus(healthData);
      setStats(statsData.data);
    } catch (err) {
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const quickActions = [
    {
      title: 'Predict Risk',
      description: 'Get real-time risk predictions for any location',
      icon: <BarChart3 size={32} />,
      accent: '#818cf8',
      link: '/tharushi/predict',
    },
    {
      title: 'View Map',
      description: 'Explore risk areas on interactive map',
      icon: <Map size={32} />,
      accent: '#3b82f6',
      link: '/tharushi/map-calendar',
    },
    {
      title: 'Historical Data',
      description: 'Analyze past conflicts and trends',
      icon: <TrendingUp size={32} />,
      accent: '#f59e0b',
      link: '/tharushi/historical',
    },
  ];

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1 }}>
        <Navbar />
        <main style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Loading text="Loading dashboard..." />
        </main>
        <Footer />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1 }}>
        <Navbar />
        <main style={{ flex: 1, maxWidth: '1400px', margin: '0 auto', padding: '32px 24px', width: '100%' }}>
          <ErrorMessage message={error} />
          <button
            onClick={loadDashboardData}
            className="btn-emerald"
            style={{ marginTop: '16px' }}
          >
            Retry
          </button>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1 }}>
      <Navbar />

      <main style={{ flex: 1, maxWidth: '1400px', margin: '0 auto', padding: '32px 24px', width: '100%' }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '32px', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h1 className="page-title-gradient" style={{ fontSize: '2rem', marginBottom: '6px' }}>
              Wildlife Conflict Dashboard
            </h1>
            <p style={{ color: '#9ca3af', fontSize: '15px' }}>Monitor and analyze wildlife conflict risks in real-time</p>
          </div>
          <button
            onClick={loadDashboardData}
            className="btn-emerald"
            style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', color: '#d1d5db', boxShadow: 'none', padding: '10px 18px', display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <RefreshCw size={18} />
            Refresh
          </button>
        </div>

        {/* Stats Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', marginBottom: '32px' }}>
          <StatCard
            title="High Risk Areas"
            value={stats?.predictions?.high_risk_count || 0}
            icon={<AlertTriangle size={28} />}
            color="danger"
          />
          <StatCard
            title="Medium Risk Areas"
            value={stats?.predictions?.medium_risk_count || 0}
            icon={<Info size={28} />}
            color="warning"
          />
          <StatCard
            title="Low Risk Areas"
            value={stats?.predictions?.low_risk_count || 0}
            icon={<CheckCircle size={28} />}
            color="primary"
          />
          <StatCard
            title="Total Predictions"
            value={stats?.predictions?.total_predictions || 0}
            icon={<BarChart3 size={28} />}
            color="info"
          />
        </div>

        {/* Quick Actions */}
        <Card title="Quick Actions" className="mb-8">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
            {quickActions.map((action, index) => (
              <Link
                key={index}
                to={action.link}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  padding: '20px',
                  borderRadius: '14px',
                  border: `1px solid rgba(255,255,255,0.1)`,
                  background: 'rgba(255,255,255,0.04)',
                  textDecoration: 'none',
                  transition: 'all 0.3s',
                  borderLeft: `4px solid ${action.accent}`,
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; e.currentTarget.style.transform = 'translateY(-2px)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.transform = 'none'; }}
              >
                <div style={{ marginBottom: '12px', color: action.accent }}>{action.icon}</div>
                <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#f3f4f6', marginBottom: '6px' }}>{action.title}</h3>
                <p style={{ fontSize: '13px', color: '#9ca3af', lineHeight: 1.5 }}>{action.description}</p>
              </Link>
            ))}
          </div>
        </Card>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '24px' }}>
          {/* System Status */}
          <Card title="System Status">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {[
                { label: 'API Status', value: systemStatus?.api || 'Running', color: 'var(--emerald-400)', dot: 'var(--emerald-500)' },
                { label: 'Model Status', value: systemStatus?.model_loaded ? 'Loaded' : 'Not Loaded', color: systemStatus?.model_loaded ? '#818cf8' : '#f87171', dot: systemStatus?.model_loaded ? '#818cf8' : '#ef4444' },
                { label: 'Data Status', value: systemStatus?.data_loaded ? 'Loaded' : 'Not Loaded', color: systemStatus?.data_loaded ? '#c084fc' : '#f87171', dot: systemStatus?.data_loaded ? '#c084fc' : '#ef4444' },
              ].map((item, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', background: 'rgba(255,255,255,0.04)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: item.dot, animation: i === 0 ? 'pulse 2s ease-in-out infinite' : undefined }} />
                    <span style={{ fontWeight: 500, color: '#d1d5db', fontSize: '14px' }}>{item.label}</span>
                  </div>
                  <span style={{ fontWeight: 700, color: item.color, fontSize: '14px' }}>{item.value}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Recent Predictions */}
          <Card title="Recent Predictions">
            {stats?.predictions?.recent_predictions?.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '280px', overflowY: 'auto' }}>
                {stats.predictions.recent_predictions.map((pred, index) => {
                  const riskColor = pred.risk_level === 'HIGH' ? '#ef4444' : pred.risk_level === 'MEDIUM' ? '#f59e0b' : 'var(--emerald-500)';
                  return (
                    <div key={index} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', paddingBottom: '10px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                      <div style={{ padding: '8px', borderRadius: '10px', background: `${riskColor}20`, color: riskColor, flexShrink: 0 }}>
                        <MapPin size={18} />
                      </div>
                      <div style={{ flex: 1 }}>
                        <p style={{ fontSize: '14px', fontWeight: 600, color: '#f3f4f6', marginBottom: '2px' }}>
                          <span style={{ color: riskColor }}>{pred.risk_level}</span> Risk &mdash; {pred.risk_score.toFixed(2)}
                        </p>
                        <p style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '2px' }}>
                          Lat: {pred.latitude.toFixed(4)}, Lon: {pred.longitude.toFixed(4)}
                        </p>
                        <p style={{ fontSize: '11px', color: '#6b7280' }}>{formatDate(pred.timestamp)}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p style={{ textAlign: 'center', color: '#6b7280', padding: '32px 0', fontSize: '14px' }}>No recent predictions</p>
            )}
          </Card>
        </div>
      </main>

      <Footer />
    </div>
  );
}

