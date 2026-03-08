import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function OfficerLogin() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const nav = useNavigate();

  async function login(e) {
    if (e) e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:8000/api/auth/login", {
        email,
        password,
      });
      localStorage.setItem("officer_token", response.data.token);
      localStorage.setItem("officer_user", JSON.stringify(response.data.user));
      nav("/officer/dashboard");
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === "Enter") login();
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-icon">🛡️</div>
        <h1 className="login-title">Officer Login</h1>
        <p className="login-subtitle">Restricted access for wildlife officers</p>

        {error && <div className="error">{error}</div>}

        <form className="login-form" onSubmit={login}>
          <div className="form-group">
            <label>👤 Officer Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="glass-input"
              placeholder="Enter your email"
              autoFocus
            />
          </div>
          <div className="form-group" style={{ marginTop: "1rem" }}>
            <label>🔒 Officer Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="glass-input"
              placeholder="Enter your password"
            />
          </div>
          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? "Authenticating..." : "Login"}
          </button>
        </form>
      </div>
    </div>
  );
}
