import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import Home from './components/Home';
import CameraView from './components/CameraView';

function App() {
  return (
    <Router>
      <div className="container mt-3">
        <nav>
          <ul className="nav">
            <li className="nav-item">
              <Link to="/" className="nav-link">Home</Link>
            </li>
            <li className="nav-item">
              <Link to="/camera-view" className="nav-link">Camera View</Link>
            </li>
          </ul>
        </nav>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/camera-view" element={<CameraView />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
