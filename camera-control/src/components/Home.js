import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

function Home() {
    const [cameraNumber, setCameraNumber] = useState('');
    const [rtspUrl, setRtspUrl] = useState('');
    const [message, setMessage] = useState('');
    const [messageType, setMessageType] = useState('');

    const startCamera = async () => {
        try {
            const response = await axios.post('http://localhost:5000/start-camera', {
                cameraNumber,
                rtspUrl
            });
            setMessage(response.data.message);
            setMessageType('success');
        } catch (error) {
            if (error.response) {
                setMessage(error.response.data.message);
                setMessageType('danger');
            } else if (error.request) {
                setMessage('No response received from the server.');
                setMessageType('danger');
            } else {
                setMessage('Error in setting up the request.');
                setMessageType('danger');
            }
        }
    };

    const stopCamera = async () => {
        try {
            const response = await axios.post('http://localhost:5000/stop-camera', {
                cameraNumber
            });
            setMessage(response.data.message);
            setMessageType('success');
        } catch (error) {
            if (error.response) {
                setMessage(error.response.data.message);
                setMessageType('danger');
            } else if (error.request) {
                setMessage('No response received from the server.');
                setMessageType('danger');
            } else {
                setMessage('Error in setting up the request.');
                setMessageType('danger');
            }
        }
    };

    const showMessage = (message, type) => {
        return (
            <div className={`alert alert-${type}`} role="alert">
                {message}
            </div>
        );
    };

    return (
        <div className="container mt-5">
            <h1 className="text-center">Camera Control</h1>
            <div className="row justify-content-center">
                <div className="col-md-6">
                    <form id="cameraForm">
                        <div className="mb-3">
                            <label htmlFor="cameraNumber" className="form-label">Camera Number</label>
                            <input 
                                type="number" 
                                className="form-control" 
                                id="cameraNumber" 
                                value={cameraNumber} 
                                onChange={(e) => setCameraNumber(e.target.value)} 
                                required 
                            />
                        </div>
                        <div className="mb-3">
                            <label htmlFor="rtspUrl" className="form-label">RTSP URL</label>
                            <input 
                                type="url" 
                                className="form-control" 
                                id="rtspUrl" 
                                value={rtspUrl} 
                                onChange={(e) => setRtspUrl(e.target.value)} 
                                required 
                            />
                        </div>
                        <div className="d-grid gap-2">
                            <button type="button" className="btn btn-success" onClick={startCamera}>Start Camera</button>
                            <button type="button" className="btn btn-danger" onClick={stopCamera}>Stop Camera</button>
                        </div>
                    </form>
                </div>
            </div>
            <div className="row justify-content-center mt-4">
                <div className="col-md-6">
                    {message && showMessage(message, messageType)}
                </div>
            </div>
        </div>
    );
}

export default Home;
