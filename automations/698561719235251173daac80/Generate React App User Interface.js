// This step scaffolds a React app UI for interacting with the FastAPI churn model backend.
// Place these files in a standard create-react-app project directory (or run `npx create-react-app ml-frontend` then copy these over).
// --
// The backend API URL is configured via BASE_URL below or environment (see .env.example)

// ========== package.json ==========
{
  "name": "ml-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.6.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  }
}

// ========== .env.example ==========
# Copy/rename to .env for local development
REACT_APP_BACKEND_BASE_URL=http://localhost:8000

// ========== src/App.js ==========
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BASE_URL = process.env.REACT_APP_BACKEND_BASE_URL || 'http://localhost:8000';

function App() {
  const [featureNames, setFeatureNames] = useState([]);
  const [form, setForm] = useState({});
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    // Fetch available features from backend status endpoint
    async function fetchFeatures() {
      try {
        const resp = await axios.get(`${BASE_URL}/`);
        // Allow both array-form and info object
        if (resp.data && resp.data.features) {
          setFeatureNames(resp.data.features);
          setForm(Object.fromEntries(resp.data.features.map(f => [f, ''])));
        } else {
          // Fallback static example
          setFeatureNames(['feature1', 'feature2']);
          setForm({ feature1: '', feature2: '' });
        }
        setStatus(resp.data.status || 'Ready');
      } catch (e) {
        setError('Could not connect to API. Check the backend URL.');
        setFeatureNames(['feature1', 'feature2']);
        setForm({ feature1: '', feature2: '' });
      }
    }
    fetchFeatures();
  }, []);

  const handleChange = (e, name) => {
    setForm({ ...form, [name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError('');
    try {
      // POST to /predict
      const resp = await axios.post(`${BASE_URL}/predict`, form);
      setResult(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Check input format & backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: '2rem auto', fontFamily: 'Arial, sans-serif' }}>
      <h2>ML Model Prediction UI</h2>
      <div><b>Status:</b> {status}</div>
      <form onSubmit={handleSubmit} style={{ marginTop: '1rem' }}>
        {featureNames.map(name => (
          <div key={name} style={{ margin: '8px 0' }}>
            <label>{name}:&nbsp;</label>
            <input type="text" value={form[name] || ''} onChange={e => handleChange(e, name)} style={{ width: 200 }} required />
          </div>
        ))}
        <button type="submit" disabled={loading}>Predict</button>
      </form>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}
      {result && (
        <div style={{ marginTop: 24 }}>
          <b>Prediction Result:</b>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;

// ========== src/index.js ==========
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);

// ========== public/index.html ==========
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ML Model Prediction UI</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>

// ========== README (setup instructions) ==========
# React Frontend UI for Churn Model

1. `cd ml-frontend`
2. `npm install`
3. Copy `.env.example` to `.env` and set the backend URL as needed
4. `npm start` (runs at http://localhost:3000)

> Connects to backend API at the URL in `REACT_APP_BACKEND_BASE_URL` (default: http://localhost:8000)
