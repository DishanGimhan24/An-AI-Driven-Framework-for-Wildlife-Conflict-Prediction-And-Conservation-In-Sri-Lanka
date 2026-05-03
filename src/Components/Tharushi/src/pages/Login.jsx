import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login as loginAPI } from '../api/authAPI';
import Button from '../components/common/Button';
import ErrorMessage from '../components/common/ErrorMessage';
import Input from '../components/common/Input';
import { useApp } from '../context/AppContext';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const { login } = useApp();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!username || !password) {
      setError('Please enter username and password');
      return;
    }

    setLoading(true);

    try {
      const response = await loginAPI(username, password);
      
      if (response.status === 'success') {
        login(response.data.user);
        navigate('/tharushi/dashboard');
      } else {
        setError('Invalid credentials');
      }
    } catch (err) {
      setError(err.message || 'Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '16px',
      position: 'relative',
      zIndex: 1,
    }}>
      <div style={{ maxWidth: '440px', width: '100%' }}>
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '80px',
            height: '80px',
            background: 'linear-gradient(135deg, var(--emerald-600), var(--emerald-800))',
            borderRadius: '50%',
            fontSize: '36px',
            marginBottom: '16px',
            boxShadow: '0 0 30px rgba(16,185,129,0.4)',
            animation: 'float 3s ease-in-out infinite',
          }}>
            🐘
          </div>
          <h1 style={{
            fontSize: '2rem',
            fontWeight: 800,
            background: 'linear-gradient(135deg, #ffffff 0%, var(--emerald-400) 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            marginBottom: '8px',
          }}>ELESAFE</h1>
          <p style={{ color: '#9ca3af', fontSize: '14px' }}>Wildlife Conflict Prediction System</p>
        </div>

        {/* Login Card */}
        <div className="glass-card">
          <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--emerald-400)', marginBottom: '24px' }}>
            Sign In
          </h2>

          <ErrorMessage message={error} onClose={() => setError('')} />

          <form onSubmit={handleSubmit}>
            <Input
              label="Username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              required
            />

            <Input
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
            />

            <Button
              type="submit"
              variant="primary"
              className="w-full"
              loading={loading}
              disabled={loading}
              style={{ width: '100%', marginTop: '8px' }}
            >
              Login
            </Button>
          </form>

          {/* Demo credentials */}
          <div style={{
            marginTop: '24px',
            padding: '14px 16px',
            background: 'rgba(16,185,129,0.08)',
            border: '1px solid rgba(16,185,129,0.25)',
            borderRadius: '12px',
          }}>
            <p style={{ fontSize: '13px', fontWeight: 600, color: 'var(--emerald-400)', marginBottom: '6px' }}>Demo Credentials</p>
            <p style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '2px' }}>
              Username: <span style={{ fontFamily: 'monospace', color: '#e5e7eb' }}>admin</span>
            </p>
            <p style={{ fontSize: '12px', color: '#9ca3af' }}>
              Password: <span style={{ fontFamily: 'monospace', color: '#e5e7eb' }}>password</span>
            </p>
          </div>
        </div>

        <div style={{ textAlign: 'center', marginTop: '20px' }}>
          <p style={{ fontSize: '12px', color: '#6b7280' }}>© 2025 ELESAFE — Protecting Communities &amp; Wildlife</p>
        </div>
      </div>
    </div>
  );
}