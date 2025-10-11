import { useEffect, useState } from "react";

export default function FileSelect({ type, onSelect }) {
  const [files, setFiles] = useState([]);

  useEffect(() => {
    if (!type) return;
    fetch("http://localhost:8000/api/datasets")
      .then((res) => res.json())
      .then((json) => {
        setFiles(json[type] || []);
        if (json[type]?.length) onSelect(json[type][0]);
      })
      .catch((err) => console.error(err));
  }, [type, onSelect]);

  return (
    <select onChange={(e) => onSelect(e.target.value)}>
      {files.map((file, i) => (
        <option key={i} value={file}>
          {file}
        </option>
      ))}
    </select>
  );
}
