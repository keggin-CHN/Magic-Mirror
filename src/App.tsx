import { LaunchPage } from "@/pages/Launch";
import { LoginPage } from "@/pages/Login";
import { MirrorPage } from "@/pages/Mirror";
import { Route, BrowserRouter as Router, Routes } from "react-router-dom";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LaunchPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/mirror" element={<MirrorPage />} />
      </Routes>
    </Router>
  );
}

export default App;
