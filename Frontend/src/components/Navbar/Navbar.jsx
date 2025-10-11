import { useLocation } from "react-router-dom";
import "./navbar.css";

export default function Navbar() {
  const location = useLocation();

  let options = [];

  if (location.pathname === "/medical") {
    options = ["ECG", "EEG"];
  } else if (location.pathname === "/sound") {
    options = ["Radar", "Doppler"];
  }

  return (
    <nav>
      <Select name="mode" data={options} />

      <div>
        <Select name="lead" />
        <Select />
      </div>
    </nav>
  );
}

function Select({ data = [], name = "option", size = 3, defaultValue = "" }) {
  const options =
    data.length > 0
      ? data
      : Array.from({ length: size }, (_, i) => `${name} ${i + 1}`);

  return (
    <select defaultValue={defaultValue}>
      <option value="" disabled>
        Select {name}
      </option>
      {options.map((item, i) => (
        <option key={i} value={item}>
          {item}
        </option>
      ))}
    </select>
  );
}
