import { useState, useEffect, useRef } from "react";
import Plot from "react-plotly.js";
import FileSelect from "../FileSelect/FileSelect";
import Button from "../Button/Button";
import "./Page.css";

export default function Page({ children }) {
  const [selectedFile, setSelectedFile] = useState("");
  const [type, setType] = useState(""); // ecg / eeg / sound
  const [fullData, setFullData] = useState([]); // بيانات كاملة: array of channels
  const [displayData, setDisplayData] = useState([]); // بيانات معروضة: array of channels
  const [isRunning, setIsRunning] = useState(false);
  const pointerRef = useRef(0);
  const windowSize = 500; // حجم النافذة المعروضة

  // تحديد النوع حسب النص في children
  useEffect(() => {
    if (children.includes("ECG")) setType("ecg");
    else if (children.includes("EEG")) setType("eeg");
    else if (children.includes("Sound")) setType("sound");
  }, [children]);

  // جلب البيانات عند اختيار الملف
  useEffect(() => {
    if (!selectedFile || !type) return;
    const filename = selectedFile.split(".")[0];
    fetch(`http://localhost:8000/api/${type}/${filename}`)
      .then((res) => res.json())
      .then((json) => {
        let data = json.data;
        // التأكد إن كل شيء array of channels
        if (Array.isArray(data[0])) setFullData(data);
        else setFullData([data]); // single channel يصبح channel واحد
        pointerRef.current = 0;
        setDisplayData([]);
      })
      .catch((err) => console.error(err));
  }, [selectedFile, type]);

  // تحديث البيانات المعروضة (sliding)
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      const ptr = pointerRef.current;
      const nextPtr = ptr + 5; // عدد العينات المضافة في كل مرة

      const newDisplay = fullData.map((channel) => {
        const slice = channel.slice(ptr, nextPtr);
        return slice;
      });

      pointerRef.current = nextPtr >= fullData[0].length ? 0 : nextPtr;

      setDisplayData((prev) => {
        if (prev.length === 0) return newDisplay;
        return prev.map((channel, idx) => {
          const combined = [...channel, ...newDisplay[idx]];
          return combined.slice(-windowSize);
        });
      });
    }, 50);

    return () => clearInterval(interval);
  }, [isRunning, fullData]);

  const start = () => setIsRunning(true);
  const pause = () => setIsRunning(false);
  const stop = () => {
    setIsRunning(false);
    pointerRef.current = 0;
    setDisplayData([]);
  };

  const colors = ["blue", "red", "green", "orange", "purple", "cyan"]; // ألوان للقنوات

  return (
    <div className="medical-container">
      <div className="monitor-section">
        {children}
        {type && <FileSelect type={type} onSelect={setSelectedFile} />}
        {displayData.length > 0 && (
          <Plot
            data={displayData.map((channel, idx) => ({
              y: channel,
              type: "scatter",
              mode: "lines",
              name: `Channel ${idx + 1}`,
              line: { color: colors[idx % colors.length] },
            }))}
            layout={{ width: 900, height: 400, title: selectedFile }}
          />
        )}
      </div>

      <div className="btn-container">
        <Button bgColor="#4da6ff" onClick={start}>
          Start
        </Button>
        <Button bgColor="#89cff0" onClick={pause}>
          Pause
        </Button>
        <Button bgColor="#ff4d6d" onClick={stop}>
          Stop
        </Button>
      </div>
    </div>
  );
}
