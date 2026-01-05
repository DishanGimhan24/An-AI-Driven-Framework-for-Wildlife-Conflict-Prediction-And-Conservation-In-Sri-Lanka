import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import ElephantMap from './ElephantMap';
import PredictPage from './PredictPage';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<ElephantMap />} />
          <Route path="/predict" element={<PredictPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
