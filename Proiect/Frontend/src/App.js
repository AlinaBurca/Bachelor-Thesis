import React, { useRef, useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState('');
  const [procent, setPercent] = useState(0);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(URL.createObjectURL(e.target.files[0]));
      setFile(e.target.files[0]);
      setResult('');
    }
  };

  const handleRemoveImage = () => {
    setImage(null);
    setFile(null);
    setResult('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleChooseClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setResult('');
    const formData = new FormData();
    formData.append('image', file);
    console.log('Form data prepared:', formData.get('image'));

    try {
       
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log(data);
      setResult(data.prediction || 'Eroare la predicție');
      setPercent(data.procentage || 0);
      if (data.procentage) {
        console.log('Procentaj:', data.procentage);
      } else {
        console.error('Procentajul nu a fost returnat de server');
      }
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      if (data.error) {
        setResult(data.error);
      }
    } catch (error) {
      setResult('Eroare la conectarea cu serverul');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h2>AlzDetect: Alzheimer stage disease classifier</h2>
      </header>
      <main className="App-content">
        <div className="top-section">
          <div className="illustration">
            <img
              src={process.env.PUBLIC_URL + '/image.png'}
              alt="Ilustrație"
              style={{ maxWidth: '100%', maxHeight: '400px' }}
            />
          </div>
          <div className="input-section">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              ref={fileInputRef}
              style={{ display: 'none' }}
            />
            {!image && (
              <button className="choose-btn" onClick={handleChooseClick}>Upload image</button>
            )}
           {image ? (
            <>
              <img
                src={image}
                alt="Poza încărcată"
                style={{ maxWidth: '100%', maxHeight: '300px', marginTop: 20, borderRadius: 8 }}
              />
              <button onClick={handleRemoveImage} style={{ marginTop: 10 }}>Delete image</button>
              {!result && (
                <button
                  className="predict-btn"
                  onClick={handlePredict}
                  style={{ marginTop: 10 }}
                  disabled={loading}
                >
                  {loading ? 'Waiting for the result' : 'Predict'}
                </button>
              )}
              {result && (
                <div className="result" style={{ marginTop: 15, color: '#023e8a', fontWeight: 600 }}>
                  Predicted stage: <strong>{result}</strong> with <strong>{procent}%</strong> confidence
                </div>
              )}
            </>
          ) : (
            <div className="placeholder" style={{ marginTop: 20 }}>No image uploaded</div>
          )}
          </div>
        </div>

        {/* Alzheimer stages section */}
        <div className="Alzheimer-stages">
          <h3>Alzheimer stages</h3>
          <div className="stages-list">
            <div className="stage-column">
              <img src={process.env.PUBLIC_URL + '/NonDemented.jpg'} alt="Stage 1" className="stage-img" />
              <h4>Stage 1: Non Demented</h4>
            </div>
            <div className="stage-column">
              <img src={process.env.PUBLIC_URL + '/VeryMild.jpg'} alt="Stage 2" className="stage-img" />
              <h4>Stage 2: VeryMild Demented</h4>
           
            </div>
            <div className="stage-column">
              <img src={process.env.PUBLIC_URL + '/Mild.jpg'} alt="Stage 3" className="stage-img" />
              <h4>Stage 3: Mild Demented</h4>
            </div>
            <div className="stage-column">
              <img src={process.env.PUBLIC_URL + '/ModerateDemented.jpg'} alt="Stage 4" className="stage-img" />
              <h4>Stage 4: Moderate Demented</h4>
            
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;