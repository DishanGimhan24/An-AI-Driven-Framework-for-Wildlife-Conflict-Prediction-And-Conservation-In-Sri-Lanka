import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="home-container">
      <div className="home-hero">
        <h1>🐘 Wildlife Safety System</h1>
        <p>
          Community reporting for citizens and an operational dashboard for wildlife officers.
          Protecting Sri Lanka's precious wildlife through AI-powered early warning systems.
        </p>
      </div>

      <div className="home-cards">
        <div className="home-card">
          <div className="home-card-icon">📝</div>
          <h2>Community Reporting</h2>
          <p>Report illegal poaching or wildlife offences. Quick and simple. Your report helps protect wildlife.</p>
          <Link className="home-card-btn" to="/report">Make a Report</Link>
        </div>

        <div className="home-card">
          <div className="home-card-icon">🛡️</div>
          <h2>Officer Portal</h2>
          <p>View reports, track status, and check predicted risk. Restricted access for wildlife officers.</p>
          <Link className="home-card-btn" to="/officer/login">Officer Login</Link>
        </div>
      </div>
    </div>
  );
}
