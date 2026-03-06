import { Link } from "react-router-dom";
import "./Navbar.css";

export default function Navbar() {
  return (
    <nav className="nav">
      <div className="nav__left">
        <h2 className="nav__logo">AI-AVC</h2>
        <span className="nav__tag">Wildlife Safety Platform</span>
      </div>

      <div className="nav__right">
        <ul className="nav__menu">
          <li><Link to="/">Home</Link></li>
          <li><Link to="/dashboard">Dashboard</Link></li>
        </ul>

        <div className="nav__profile">
          <span className="nav__statusDot" title="Online"></span>
          <img
            className="nav__avatar"
            src="https://i.pravatar.cc/80?img=12"
            alt="Profile"
          />
        </div>
      </div>
    </nav>
  );
}