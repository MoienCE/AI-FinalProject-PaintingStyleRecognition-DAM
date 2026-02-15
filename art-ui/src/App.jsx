import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Send to Flask API
      const response = await axios.post('http://127.0.0.1:5000/predict', formData);
      setResult(response.data);
    } catch (err) {
      setError("Failed to connect to the AI model. Is the backend running?");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>🎨 Art Style Identifier</h1>
        <p>Upload a painting to discover its artistic movement</p>
      </header>

      <div className="main-content">
        <div className="upload-section">
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange} 
            id="file-upload"
            className="hidden-input"
          />
          <label htmlFor="file-upload" className="upload-btn">
            {preview ? "Choose Another Image" : "Upload Painting"}
          </label>

          {preview && (
            <div className="image-preview">
              <img src={preview} alt="Preview" />
            </div>
          )}

          {file && (
            <button 
              onClick={handleUpload} 
              disabled={loading} 
              className="analyze-btn"
            >
              {loading ? "Analyzing..." : "🔍 Identify Style"}
            </button>
          )}
          
          {error && <p className="error-msg">{error}</p>}
        </div>

        {result && (
          <div className="result-section">
            <div className="top-result">
              <h2>It looks like...</h2>
              <div className="badge">{result.top_prediction.style}</div>
              <div className="confidence-bar-container">
                <div 
                  className="confidence-bar" 
                  style={{width: `${result.top_prediction.confidence}%`}}
                ></div>
              </div>
              <p className="confidence-text">{result.top_prediction.confidence}% Confidence</p>
            </div>

            <div className="alternatives">
              <h3>Other Possibilities:</h3>
              {result.alternatives.map((alt, index) => (
                <div key={index} className="alt-item">
                  <span>{alt.style}</span>
                  <span>{alt.confidence}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;