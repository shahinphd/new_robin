import React, { useEffect, useState } from 'react';

const CameraView = () => {
  const [frames, setFrames] = useState({});
  useEffect(() => {
    const cameras = [1, 2, 3, 4]; // Assuming you have 4 cameras
    const wsConnections = cameras.map(camId => {
      const ws = new WebSocket(`ws://localhost:8765/${camId}`);
      ws.onmessage = (event) => {
        const frame = event.data;
        setFrames(prevFrames => ({ ...prevFrames, [camId]: frame }));
      };
      ws.onclose = () => {
        console.log(`WebSocket for Camera ${camId} closed`);
      };
      return ws;
    });

    return () => {
      wsConnections.forEach(ws => ws.close());
    };
  }, []);

  return (
    <div className="camera-view">
      <div className="camera-grid">
        {Array.from({ length: 4 }).map((_, index) => (
          <div className="camera-box" key={index}>
            <h3>Camera {index + 1}</h3>
            {frames[index + 1] ? (
              <img src={`data:image/jpeg;base64,${frames[index + 1]}`} alt={`Camera ${index + 1}`} />
            ) : (
              <p>No frame received</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default CameraView;
