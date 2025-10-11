import { useState } from "react";
import Button from "../Button/Button";
import "./Intro.css";

export default function Intro() {
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen) return null;

  return (
    <div className="wrapper">
      <div className="modal">
        <button className="close-btn" onClick={() => setIsOpen(false)}>
          ×
        </button>
        <h1>Which Mode do you want?</h1>
        <div className="modal-buttons">
          <Button bgColor="#1e6091" to="http://127.0.0.1:8052/">
            Medical
          </Button>
          {/* <Button bgColor="#5ca8d8" to={"/ecg"}>
            ECG
          </Button> */}
          <Button bgColor="#28a745" to="http://127.0.0.1:8050/">
            Doppler
          </Button>
          <Button bgColor="#ff6b6b" to={"http://127.0.0.1:8053/"}>
            SAR
          </Button>
          <Button bgColor="#9d4edd" to="http://127.0.0.1:8051/">
            Sound
          </Button>
        </div>
      </div>
    </div>
  );
}
