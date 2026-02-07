import React, { useState } from 'react';
import { Activity, DollarSign, Calendar, CreditCard, AlertTriangle, CheckCircle, User, Server } from 'lucide-react';
import './index.css';

const App = () => {
  const [formData, setFormData] = useState({
    tenure: 12,
    MonthlyCharges: 70.50,
    TotalCharges: 840.00,
    Contract: 'Month-to-month',
    PaymentMethod: 'Electronic check'
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'tenure' ? parseInt(value) : 
              name.includes('Charges') ? parseFloat(value) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // UPDATED: Pointing to Port 8000
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: formData })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError('Failed to connect to backend (Port 8000). Is python src/app.py running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="main-layout">
        
        {/* Left Column: Input Form */}
        <div className="card form-card">
          <div className="card-header">
            <div className="icon-box primary">
              <Activity className="icon" />
            </div>
            <div>
              <h1>Churn Predictor</h1>
              <div className="status-badge">
                <span className="status-dot"></span>
                <span className="status-text">System Online</span>
              </div>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="form-content">
            
            {/* Tenure Input */}
            <div className="input-group range-group">
              <div className="label-row">
                <label><User className="label-icon" /> Tenure</label>
                <span className="value-badge">{formData.tenure} Months</span>
              </div>
              <input
                type="range"
                name="tenure"
                min="0"
                max="72"
                value={formData.tenure}
                onChange={handleChange}
                className="range-input"
              />
              <div className="range-labels">
                <span>0</span>
                <span>72</span>
              </div>
            </div>

            <div className="row">
              {/* Monthly Charges */}
              <div className="input-group">
                <label><DollarSign className="label-icon green" /> Monthly</label>
                <div className="input-wrapper">
                  <span className="currency-symbol">$</span>
                  <input
                    type="number"
                    name="MonthlyCharges"
                    step="0.01"
                    value={formData.MonthlyCharges}
                    onChange={handleChange}
                  />
                </div>
              </div>

              {/* Total Charges */}
              <div className="input-group">
                <label><DollarSign className="label-icon green" /> Total</label>
                <div className="input-wrapper">
                  <span className="currency-symbol">$</span>
                  <input
                    type="number"
                    name="TotalCharges"
                    step="0.01"
                    value={formData.TotalCharges}
                    onChange={handleChange}
                  />
                </div>
              </div>
            </div>

            {/* Select Inputs */}
            <div className="row">
              <div className="input-group">
                <label><Calendar className="label-icon orange" /> Contract</label>
                <select name="Contract" value={formData.Contract} onChange={handleChange}>
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>

              <div className="input-group">
                <label><CreditCard className="label-icon blue" /> Payment</label>
                <select name="PaymentMethod" value={formData.PaymentMethod} onChange={handleChange}>
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer</option>
                  <option value="Credit card (automatic)">Credit card</option>
                </select>
              </div>
            </div>

            <button type="submit" disabled={loading} className="submit-btn">
              {loading ? (
                <span className="loading-text"><Server className="spin" /> Processing...</span>
              ) : (
                'Run Analysis'
              )}
            </button>
          </form>
        </div>

        {/* Right Column: Visualization */}
        <div className="results-column">
          
          {/* Default State */}
          {!result && !error && (
            <div className="card empty-state">
              <div className="icon-circle">
                <Activity className="icon-large" />
              </div>
              <h3>Ready to Analyze</h3>
              <p>Enter customer metrics to generate a real-time ML inference.</p>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="card error-state">
              <AlertTriangle className="icon-large error" />
              <h3>Connection Failed</h3>
              <p>{error}</p>
              <button onClick={handleSubmit} className="retry-btn">Retry Connection</button>
            </div>
          )}

          {/* Success State */}
          {result && (
            <div className={`card result-card ${result.prediction === 'Churn' ? 'danger' : 'safe'}`}>
              
              <div className="result-header">
                <div className="result-icon-wrapper">
                  {result.prediction === 'Churn' 
                    ? <AlertTriangle className="result-icon" />
                    : <CheckCircle className="result-icon" />
                  }
                </div>
                <h2>{result.prediction === 'Churn' ? 'High Risk' : 'Retained'}</h2>
                <p className="prediction-desc">
                  {result.prediction === 'Churn' 
                    ? 'Customer is likely to cancel.' 
                    : 'Customer is likely to stay.'}
                </p>
              </div>

              <div className="meter-container">
                <div className="meter-labels">
                  <span>Confidence Score</span>
                  <span>{(result.probability * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bg">
                  <div 
                    className="progress-fill"
                    style={{ width: `${result.probability * 100}%` }}
                  ></div>
                </div>
              </div>

              <div className="metrics-grid">
                <div className="metric-box">
                  <small>PREDICTION</small>
                  <strong>{result.prediction}</strong>
                </div>
                <div className="metric-box">
                  <small>RISK LEVEL</small>
                  <strong>{result.risk_level}</strong>
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;