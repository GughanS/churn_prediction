import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx'; // Now resolves correctly to App.jsx in the same folder
import './index.css';        // Now resolves correctly to index.css in the same folder

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);