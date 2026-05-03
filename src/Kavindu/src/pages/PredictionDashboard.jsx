import PredictForm from "../components/PredictForm";
import RiskCard    from "../components/RiskCard";
import { usePrediction } from "../hooks/usePrediction";

export default function PredictionDashboard() {
  const { result, loading, error, predict, reset } = usePrediction();

  return (
    <div className="page predict-page">
      <div className="page-header">
        <h1>🔮 Offence Risk Prediction</h1>
        <p className="page-sub">
          Enter a region and month to get an AI-powered risk assessment
        </p>
      </div>

      <div className="predict-layout">
        <div className="predict-left">
          <PredictForm onSubmit={predict} loading={loading} />

          {error && (
            <div className="error-box">
              ❌ {error}
              <button onClick={reset} className="btn-sm">Dismiss</button>
            </div>
          )}

          <div className="info-box">
            <h4>📖 How to Read Results</h4>
            <ul>
              <li><strong>Risk %</strong> — probability this will be a HIGH-RISK month</li>
              <li><strong>Warning level</strong> — 🔴 CRITICAL ≥70% · 🟠 HIGH ≥45% · 🟡 MEDIUM ≥25% · 🟢 LOW</li>
              <li><strong>Offence group</strong> — most likely category of offence</li>
              <li><strong>Multi-label</strong> — all offence types likely to co-occur</li>
              <li><strong>Rainfall ⚠️</strong> — drought flag means higher poaching risk</li>
            </ul>
          </div>
        </div>

        <div className="predict-right">
          {result
            ? <RiskCard result={result} />
            : (
              <div className="empty-state">
                <div className="empty-icon">🦚</div>
                <p>
                  Fill in the form and click <strong>Predict Risk</strong>
                  <br />to see the AI assessment here.
                </p>
              </div>
            )
          }
        </div>
      </div>
    </div>
  );
}
