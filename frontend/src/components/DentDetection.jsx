import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function DentDetection() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Car details form state
  const [carDetails, setCarDetails] = useState({
    brand: 'Toyota',
    car_price_lakhs: 15
  });

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError('');
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
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // First, get the image analysis
      const analysisResponse = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
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
    }
  };

  return (
    <div className="app-container">
      <header className="hero-section text-center py-5 bg-primary text-white">
        <div className="container">
          <h1 className="display-4 mb-3">Car Damage Detection & Cost Estimation</h1>
          <p className="lead">Upload a photo of your car's damage and get an instant repair cost estimate powered by AI.</p>
        </div>
      </header>

      <main className="container py-5">
        <div className="row justify-content-center">
          <div className="col-lg-8">
            <div className="card shadow-sm">
              <div className="card-body p-4">
                <h2 className="card-title text-center mb-4">Analyze Your Car Damage</h2>
                <form onSubmit={handleSubmit}>
                  <div className="row">
                    <div className="col-md-6 mb-3">
                      <label className="form-label">Car Brand:</label>
                      <select 
                        className="form-select"
                        name="brand" 
                        value={carDetails.brand} 
                        onChange={handleCarDetailsChange}
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
                    </div>

                    <div className="col-md-6 mb-3">
                      <label className="form-label">Car Price (Lakhs):</label>
                      <input
                        type="number"
                        className="form-control"
                        name="car_price_lakhs"
                        value={carDetails.car_price_lakhs}
                        onChange={handleCarDetailsChange}
                        min="5"
                        max="100"
                        step="1"
                      />
                    </div>
                  </div>

                  <div className="mb-4">
                    <label className="form-label">Upload Damage Image:</label>
                    <input 
                      type="file" 
                      className="form-control"
                      accept="image/*" 
                      onChange={handleFileChange} 
                    />
                    <small className="text-muted">Supported formats: JPG, PNG, JPEG</small>
                  </div>

                  <div className="text-center">
                    <button 
                      type="submit" 
                      className="btn btn-primary btn-lg px-5"
                      disabled={loading}
                    >
                      {loading ? (
                        <>
                          <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                          Analyzing...
                        </>
                      ) : (
                        'Analyze Damage'
                      )}
                    </button>
                  </div>
                </form>
                {error && (
                  <div className="alert alert-danger mt-3" role="alert">
                    <i className="bi bi-exclamation-triangle-fill me-2"></i>
                    {error}
                  </div>
                )}
              </div>
            </div>

            {result && (
              <div className="card shadow-sm mt-4">
                <div className="card-body p-4">
                  <h3 className="card-title text-center mb-4">Analysis Results</h3>
                  
                  <div className="row">
                    <div className="col-md-6 mb-4">
                      <div className="card h-100">
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
                    </div>

                    <div className="col-md-6 mb-4">
                      <div className="card h-100">
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
                  </div>

                  <div className="text-center mt-3">
                    <button 
                      className="btn btn-success btn-lg px-5"
                      onClick={() => navigate('/repair-shops')}
                    >
                      Find Nearby Repair Shops
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <section className="about-section py-5 bg-light">
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