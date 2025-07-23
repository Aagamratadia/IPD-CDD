import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './DentDetection.css';

function DentDetection() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  // Car details form state
  const [carDetails, setCarDetails] = useState({
    brand: 'Toyota',
    car_price_lakhs: 15
  });

  // Apply theme class to body when dark mode changes
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange({ target: { files: e.dataTransfer.files } });
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type.startsWith('image/')) {
        setFile(selectedFile);
    setError('');
        
        // Create preview
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreview(reader.result);
        };
        reader.readAsDataURL(selectedFile);
      } else {
        setError('Please select a valid image file (JPG, PNG, or JPEG)');
      }
    }
  };

  const handleCarDetailsChange = (e) => {
    const { name, value } = e.target;
    setCarDetails(prev => ({
      ...prev,
      [name]: name === 'car_price_lakhs' ? parseFloat(value) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image file.');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    setUploadProgress(0);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // First, get the image analysis
      const analysisResponse = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });
      
      // Then, get the cost prediction using the detected values
      const costResponse = await axios.post('http://127.0.0.1:8000/predict-cost', {
        brand: carDetails.brand,
        location: analysisResponse.data.detected_location,
        severity: analysisResponse.data.detected_severity,
        car_price_lakhs: carDetails.car_price_lakhs
      });

      setResult({
        ...analysisResponse.data,
        cost_prediction: costResponse.data
      });
    } catch (err) {
      setError('Prediction failed. Please check the server and try again.');
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className="app-container">
      <button 
        className="theme-toggle"
        onClick={toggleTheme}
        aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {isDarkMode ? '☀️' : '🌙'}
      </button>

      <header className="hero-section text-center py-5">
        <div className="container">
          <h1 className="display-4 mb-3">Car Damage Detection & Cost Estimation</h1>
          <p className="lead">Upload a photo of your car's damage and get an instant repair cost estimate powered by AI.</p>
        </div>
      </header>

      <main className="container py-5">
        <div className="row justify-content-center">
          <div className="col-lg-8">
            <div className="card">
              <div className="card-body p-4">
                <h2 className="card-title text-center mb-4">Analyze Your Car Damage</h2>
                <form onSubmit={handleSubmit}>
                  <div className="row">
                    <div className="col-md-6 mb-3">
                      <label className="form-label" htmlFor="brand">Car Brand:</label>
                      <select 
                        id="brand"
                        className="form-select"
                        name="brand" 
                        value={carDetails.brand} 
                        onChange={handleCarDetailsChange}
                        aria-describedby="brand-help"
                      >
                        <option value="Toyota">Toyota</option>
                        <option value="Honda">Honda</option>
                        <option value="Hyundai">Hyundai</option>
                        <option value="BMW">BMW</option>
                        <option value="Mercedes">Mercedes</option>
                        <option value="Ford">Ford</option>
                        <option value="Kia">Kia</option>
                        <option value="Audi">Audi</option>
                      </select>
                      <small id="brand-help" className="form-text text-muted">Select your car's manufacturer for accurate cost estimation</small>
                    </div>

                    <div className="col-md-6 mb-3">
                      <label className="form-label" htmlFor="price">Car Price (Lakhs):</label>
                      <input
                        id="price"
                        type="number"
                        className="form-control"
                        name="car_price_lakhs"
                        value={carDetails.car_price_lakhs}
                        onChange={handleCarDetailsChange}
                        min="5"
                        max="100"
                        step="1"
                        aria-describedby="price-help"
                      />
                      <small id="price-help" className="form-text text-muted">Enter your car's approximate value in lakhs</small>
                    </div>
                  </div>

                  <div 
                    className={`upload-area ${dragActive ? 'drag-active' : ''} ${preview ? 'has-image' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    role="button"
                    tabIndex="0"
                    aria-label="Upload damage image"
                  >
                    <input 
                      ref={fileInputRef}
                      type="file" 
                      className="visually-hidden"
                      accept="image/*" 
                      onChange={handleFileChange} 
                      aria-label="Upload damage image"
                    />
                    
                    {preview ? (
                      <div className="preview-container">
                        <img src={preview} alt="Damage preview" className="preview-image" />
                        <button 
                          type="button"
                          className="change-image-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            setFile(null);
                            setPreview(null);
                          }}
                        >
                          Change Image
                        </button>
                      </div>
                    ) : (
                      <div className="upload-prompt">
                        <div className="upload-icon">
                          <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                          </svg>
                        </div>
                        <p>Drag and drop your image here, or click to select</p>
                    <small className="text-muted">Supported formats: JPG, PNG, JPEG</small>
                      </div>
                    )}
                  </div>

                  {uploadProgress > 0 && (
                    <div className="progress mb-3">
                      <div 
                        className="progress-bar" 
                        role="progressbar" 
                        style={{ width: `${uploadProgress}%` }}
                        aria-valuenow={uploadProgress} 
                        aria-valuemin="0" 
                        aria-valuemax="100"
                      >
                        {uploadProgress}%
                      </div>
                    </div>
                  )}

                  <div className="text-center">
                    <button 
                      type="submit" 
                      className="btn btn-primary btn-lg px-5"
                      disabled={loading || !file}
                    >
                      {loading ? (
                        <>
                          <span className="loading-spinner" role="status" aria-hidden="true"></span>
                          <span>Analyzing...</span>
                        </>
                      ) : (
                        'Analyze Damage'
                      )}
                    </button>
                  </div>
                </form>

                {error && (
                  <div className="error-message" role="alert">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10"/>
                      <line x1="12" y1="8" x2="12" y2="12"/>
                      <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    {error}
                  </div>
                )}
              </div>
            </div>

            {result && (
              <div className="result-container">
                <h3 className="result-title">Analysis Results</h3>
                  
                <div className="result-grid">
                  <div className="card">
                        <div className="card-body">
                          <h4 className="card-title h5 mb-3">Damage Analysis</h4>
                          <div className="mb-3">
                            <p className="mb-2"><strong>Location:</strong> {result.detected_location}</p>
                            <p className="mb-2"><strong>Severity:</strong> {result.detected_severity}</p>
                          </div>
                          {result.yolo_result_image_base64 && (
                            <div className="mt-3">
                              <h5 className="h6 mb-2">Detection Result</h5>
                              <img
                                src={`data:image/jpeg;base64,${result.yolo_result_image_base64}`}
                                alt="Detection Result"
                                className="img-fluid rounded"
                              />
                            </div>
                          )}
                      </div>
                    </div>

                  <div className="card">
                        <div className="card-body">
                          <h4 className="card-title h5 mb-3">Cost Estimation</h4>
                          <div className="mb-3">
                            <p className="mb-2"><strong>Estimated Cost:</strong> ₹{result.cost_prediction.estimated_cost}</p>
                            <p className="mb-2"><strong>Brand:</strong> {result.cost_prediction.brand}</p>
                            <p className="mb-2"><strong>Location:</strong> {result.cost_prediction.location}</p>
                            <p className="mb-2"><strong>Severity:</strong> {result.cost_prediction.severity}</p>
                            <p className="mb-2"><strong>Car Price:</strong> ₹{result.cost_prediction.car_price_lakhs}L</p>
                        </div>
                      </div>
                    </div>
                  </div>

                <div className="text-center mt-4">
                    <button 
                      className="btn btn-success btn-lg px-5"
                      onClick={() => navigate('/repair-shops')}
                    >
                      Find Nearby Repair Shops
                    </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <section className="about-section py-5">
        <div className="container">
          <div className="row justify-content-center">
            <div className="col-lg-8 text-center">
              <h2 className="mb-4">About AutoCare AI</h2>
              <p className="lead">
                AutoCare AI leverages state-of-the-art deep learning and computer vision to help car owners quickly assess damage and estimate repair costs. Our mission is to make car repair transparent, fast, and accessible for everyone.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default DentDetection; 