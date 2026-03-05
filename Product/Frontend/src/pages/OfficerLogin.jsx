import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";

export default function OfficerLogin() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const nav = useNavigate();

  async function login() {
    setError("");
    setLoading(true);
    try {
      const data = await api.officerLogin({ password });
      localStorage.setItem("officer_token", data.token);
      nav("/officer/dashboard");
    } catch (e) {
      setError(e.message);
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

        <div className="login-form">
          <div className="form-group">
            <label>🔒 Officer Password</label>
            <input 
              type="password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)}
              onKeyPress={handleKeyPress}
              className="glass-input"
              placeholder="Enter your password"
              autoFocus
            />
          </div>
          <button className="submit-btn" onClick={login} disabled={loading}>
            {loading ? "Authenticating..." : "Login"}
          </button>
        </div>
      </div>
    </div>
  );
}
