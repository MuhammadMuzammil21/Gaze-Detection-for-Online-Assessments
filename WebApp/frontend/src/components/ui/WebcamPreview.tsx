import React from "react";
import Webcam from "react-webcam";

const videoConstraints = {
  width: 160,
  height: 120,
  facingMode: "user",
};

const WebcamPreview = () => {
  return (
    <div className="fixed bottom-4 right-4 border border-gray-300 rounded shadow-md z-50">
      <Webcam
        audio={false}
        videoConstraints={videoConstraints}
        screenshotFormat="image/jpeg"
        className="rounded"
      />
    </div>
  );
};

export default WebcamPreview;
