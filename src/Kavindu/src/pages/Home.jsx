import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Eye, Smartphone, ShieldCheck, PhoneCall, TrendingUp, Leaf, ShieldAlert, Scissors, Axe, TriangleAlert } from "lucide-react";

const AnimatedCounter = ({ end, duration = 2000, suffix = "" }) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const easeProgress = 1 - Math.pow(1 - progress, 4); // easeOutQuart
      setCount(Math.floor(easeProgress * end));
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    window.requestAnimationFrame(step);
  }, [end, duration]);

  return <span>{count.toLocaleString()}{suffix}</span>;
};

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

      {/* Live Impact Counters Section */}
      <div style={{ maxWidth: "1000px", margin: "-2rem auto 4rem", padding: "0 1rem", position: "relative", zIndex: 10 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1.5rem" }}>

          <div className="glass-card" style={{ padding: "1.5rem", textAlign: "center", borderTop: "4px solid #818cf8" }}>
            <TrendingUp size={28} color="#818cf8" style={{ margin: "0 auto 10px" }} />
            <div style={{ fontSize: "2.5rem", fontWeight: "bold", color: "white", marginBottom: "0.5rem" }}>
              <AnimatedCounter end={1240} suffix="+" />
            </div>
            <div style={{ color: "#aaa", fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "1px" }}>Reports Submitted</div>
          </div>

          <div className="glass-card" style={{ padding: "1.5rem", textAlign: "center", borderTop: "4px solid #f87171" }}>
            <ShieldAlert size={28} color="#f87171" style={{ margin: "0 auto 10px" }} />
            <div style={{ fontSize: "2.5rem", fontWeight: "bold", color: "white", marginBottom: "0.5rem" }}>
              <AnimatedCounter end={85} />
            </div>
            <div style={{ color: "#aaa", fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "1px" }}>Arrests Assisted by AI</div>
          </div>

          <div className="glass-card" style={{ padding: "1.5rem", textAlign: "center", borderTop: "4px solid #4ade80" }}>
            <Leaf size={28} color="#4ade80" style={{ margin: "0 auto 10px" }} />
            <div style={{ fontSize: "2.5rem", fontWeight: "bold", color: "white", marginBottom: "0.5rem" }}>
              <AnimatedCounter end={300} suffix="+" />
            </div>
            <div style={{ color: "#aaa", fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "1px" }}>Animals Saved</div>
          </div>

        </div>
      </div>

      <div className="how-it-works-section" style={{ padding: "3rem 1rem", textAlign: "center", marginBottom: "2rem" }}>
        <h2 style={{ fontSize: "2rem", marginBottom: "2rem", color: "var(--text-color)" }}>How It Works</h2>
        <div style={{ display: "flex", gap: "2rem", justifyContent: "center", flexWrap: "wrap", maxWidth: "900px", margin: "0 auto" }}>

          <div className="glass-card" style={{ flex: "1 1 250px", padding: "2rem", textAlign: "center" }}>
            <div style={{ background: "rgba(99,102,241,0.15)", width: "64px", height: "64px", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1.5rem", color: "#818cf8" }}>
              <Eye size={32} />
            </div>
            <h3 style={{ fontSize: "1.25rem", marginBottom: "1rem", color: "white" }}>1. You Spot It</h3>
            <p style={{ color: "#aaa", fontSize: "0.95rem", lineHeight: "1.6" }}>
              Observe suspected poaching, illegal logging, traps, or distressed wildlife happening near you.
            </p>
          </div>

          <div className="glass-card" style={{ flex: "1 1 250px", padding: "2rem", textAlign: "center" }}>
            <div style={{ background: "rgba(239,68,68,0.15)", width: "64px", height: "64px", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1.5rem", color: "#f87171" }}>
              <Smartphone size={32} />
            </div>
            <h3 style={{ fontSize: "1.25rem", marginBottom: "1rem", color: "white" }}>2. You Report It</h3>
            <p style={{ color: "#aaa", fontSize: "0.95rem", lineHeight: "1.6" }}>
              Quickly submit a confidential report with precise GPS coordinates and photo evidence right from your phone.
            </p>
          </div>

          <div className="glass-card" style={{ flex: "1 1 250px", padding: "2rem", textAlign: "center" }}>
            <div style={{ background: "rgba(34,197,94,0.15)", width: "64px", height: "64px", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1.5rem", color: "#4ade80" }}>
              <ShieldCheck size={32} />
            </div>
            <h3 style={{ fontSize: "1.25rem", marginBottom: "1rem", color: "white" }}>3. We Intervene</h3>
            <p style={{ color: "#aaa", fontSize: "0.95rem", lineHeight: "1.6" }}>
              The system immediately alerts and dispatches the closest Rapid Response Team to handle the threat.
            </p>
          </div>

        </div>
      </div>

      {/* Educational: Know What To Report */}
      <div style={{ padding: "1rem", textAlign: "center", marginBottom: "4rem" }}>
        <h2 style={{ fontSize: "2rem", marginBottom: "1rem", color: "white" }}>Know What To Report</h2>
        <p style={{ color: "#aaa", marginBottom: "2.5rem", maxWidth: "600px", margin: "0 auto 2.5rem" }}>
          Unsure if what you saw is illegal? Here are the most common offences our department handles.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1.5rem", maxWidth: "1000px", margin: "0 auto" }}>

          <div className="glass-card" style={{ padding: "1.5rem", textAlign: "left", display: "flex", gap: "1rem", alignItems: "flex-start", background: "rgba(239, 68, 68, 0.05)", borderLeft: "4px solid #ef4444" }}>
            <div style={{ background: "rgba(239, 68, 68, 0.2)", padding: "12px", borderRadius: "10px", color: "#fca5a5" }}>
              <Scissors size={24} />
            </div>
            <div>
              <h3 style={{ color: "white", fontSize: "1.1rem", marginBottom: "0.25rem" }}>Wire Snares & Traps</h3>
              <p style={{ color: "#aaa", fontSize: "0.85rem", lineHeight: "1.5", margin: 0 }}>Hidden wire loops used to catch wildlife. Often found tied to trees near watering holes or trails.</p>
            </div>
          </div>

          <div className="glass-card" style={{ padding: "1.5rem", textAlign: "left", display: "flex", gap: "1rem", alignItems: "flex-start", background: "rgba(245, 158, 11, 0.05)", borderLeft: "4px solid #f59e0b" }}>
            <div style={{ background: "rgba(245, 158, 11, 0.2)", padding: "12px", borderRadius: "10px", color: "#fcd34d" }}>
              <Axe size={24} />
            </div>
            <div>
              <h3 style={{ color: "white", fontSize: "1.1rem", marginBottom: "0.25rem" }}>Illegal Logging</h3>
              <p style={{ color: "#aaa", fontSize: "0.85rem", lineHeight: "1.5", margin: 0 }}>Unmarked trucks carrying valuable timber (like Rosewood) or sounds of chainsaws deep inside protected reserves.</p>
            </div>
          </div>

          <div className="glass-card" style={{ padding: "1.5rem", textAlign: "left", display: "flex", gap: "1rem", alignItems: "flex-start", background: "rgba(59, 130, 246, 0.05)", borderLeft: "4px solid #3b82f6" }}>
            <div style={{ background: "rgba(59, 130, 246, 0.2)", padding: "12px", borderRadius: "10px", color: "#93c5fd" }}>
              <TriangleAlert size={24} />
            </div>
            <div>
              <h3 style={{ color: "white", fontSize: "1.1rem", marginBottom: "0.25rem" }}>Distressed Wildlife</h3>
              <p style={{ color: "#aaa", fontSize: "0.85rem", lineHeight: "1.5", margin: 0 }}>Animals that are visibly injured, wandering into human settlements, or baby animals that appear orphaned.</p>
            </div>
          </div>

        </div>
      </div>

      <div className="home-cards">
        <div className="home-card">
          <div className="home-card-icon">📝</div>
          <h2>Community Reporting</h2>
          <p>Report illegal poaching or wildlife offences. Quick and simple. Your report helps protect wildlife.</p>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem" }}>
            <Link className="home-card-btn" to="/report" style={{ width: "100%" }}>Make a Report</Link>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#4ade80", background: "rgba(74, 222, 128, 0.1)", border: "1px solid rgba(74, 222, 128, 0.2)", padding: "8px 16px", borderRadius: "20px", fontSize: "0.9rem", fontWeight: "600" }}>
              <ShieldCheck size={16} />
              100% Anonymous Guarantee
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: "3rem", padding: "2rem", background: "rgba(0,0,0,0.3)", borderRadius: "16px", textAlign: "center", border: "1px solid rgba(255,255,255,0.05)", maxWidth: "600px", margin: "3rem auto 0" }}>
        <PhoneCall size={32} color="#f87171" style={{ marginBottom: "1rem" }} />
        <h2 style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>Wildlife Emergency?</h2>
        <p style={{ color: "#aaa", marginBottom: "1.5rem" }}>If human life is in immediate danger from wildlife, do not use the reporting form. Call the department directly.</p>
        <a href="tel:1992" style={{ background: "#ef4444", color: "white", padding: "12px 30px", borderRadius: "8px", fontSize: "1.2rem", fontWeight: "bold", textDecoration: "none", display: "inline-block" }}>Call 1992 Hotline</a>
      </div>

    </div>
  );
}
