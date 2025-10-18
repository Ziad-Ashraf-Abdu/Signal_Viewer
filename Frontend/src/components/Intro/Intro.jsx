import React, { useState } from "react";
import "./Intro.css";
import {
  FaHeartbeat,
  FaSatelliteDish,
  FaBroadcastTower,
  FaPlane,
  FaMicrophoneAlt,
  FaArrowRight,
} from "react-icons/fa";

// Import your new local logo from the assets folder
import teamLogo from "../../assets/attachment_135540270-removebg-preview.png";

export default function Intro() {
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const dashboardItems = [
    {
      title: "Medical",
      description: "Analyze and visualize biosignal data streams.",
      link: "http://127.0.0.1:8052/",
      icon: <FaHeartbeat />,
      color: "#1e6091",
    },
    {
      title: "Doppler",
      description: "Process and display Doppler effect simulations.",
      link: "http://127.0.0.1:8050/",
      icon: <FaBroadcastTower />,
      color: "#28a745",
    },
    {
      title: "SAR",
      description: "Interpret Synthetic-Aperture Radar outputs.",
      link: "http://127.0.0.1:8053/",
      icon: <FaSatelliteDish />,
      color: "#ff6b6b",
    },
    {
      title: "Drone",
      description: "Access telemetry and control drone simulations.",
      link: "http://127.0.0.1:8051/",
      icon: <FaPlane />,
      color: "#9d4edd",
    },
    {
      title: "Voice",
      description: "Engage with real-time audio signal processing.",
      link: "http://127.0.0.1:8054/",
      icon: <FaMicrophoneAlt />,
      color: "#fca311",
    },
  ];

  return (
      <div className="launcher-container">
        {/* LEFT COLUMN */}
        <div className="launcher-content">
          <header className="launcher-header">
            <h1>Signal Viewer</h1>
            <p>Select a simulation dashboard to begin.</p>
          </header>
          <ul className="launcher-list">
            {dashboardItems.map((item, index) => (
                <li
                    key={index}
                    onMouseEnter={() => setHoveredIndex(index)}
                    onMouseLeave={() => setHoveredIndex(null)}
                >
                  <a href={item.link} className="item-link">
                    <div className="item-icon" style={{ color: item.color }}>
                      {item.icon}
                    </div>
                    <div className="item-text">
                      <h3 className="item-title">{item.title}</h3>
                      <p className="item-description">{item.description}</p>
                    </div>
                    <div className="item-arrow">
                      <FaArrowRight />
                    </div>
                  </a>
                </li>
            ))}
          </ul>
        </div>

        {/* RIGHT COLUMN */}
        <div
            className={`preview-pane ${
                hoveredIndex !== null ? "preview-pane--active" : ""
            }`}
        >
          {/* MODIFIED: New container for the logo and text */}
          <div className={`logo-container ${hoveredIndex === null ? "visible" : ""}`}>
            <p className="logo-text">Presented from</p>
            <img
                src={teamLogo}
                alt="Team Logo"
                className="preview-logo"
            />
          </div>

          {dashboardItems.map((item, index) => (
              <iframe
                  key={index}
                  src={item.link}
                  title={`${item.title} preview`}
                  className={`preview-iframe ${
                      hoveredIndex === index ? "active" : ""
                  }`}
                  sandbox="allow-scripts allow-same-origin"
              ></iframe>
          ))}
          <div className="preview-overlay"></div>
        </div>
      </div>
  );
}