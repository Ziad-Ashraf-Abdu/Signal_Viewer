import { Link } from "react-router-dom";
import "./Button.css";

export default function Button({ children, bgColor, to }) {
  return (
    <Link to={to}>
      <button className="button" style={{ backgroundColor: bgColor }}>
        {children}
      </button>
    </Link>
  );
}
