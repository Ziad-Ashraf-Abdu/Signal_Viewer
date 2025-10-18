import { Route, Routes } from "react-router";
import Intro from "./components/Intro/Intro";
// import Navbar from "./components/Navbar/Navbar";
// import Page from "./components/Page/Page";

export default function App() {
  // const location = useLocation();

  // const showNavbar = location.pathname !== "/";
  return (
    <>
      {/* {showNavbar && <Navbar />} */}
      <Intro />
    </>
  );
}
