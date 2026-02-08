import { LaunchPage } from "@/pages/Launch";
import { MirrorPage } from "@/pages/Mirror";
import { Route, BrowserRouter as Router, Routes } from "react-router-dom";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LaunchPage />} />
        <Route path="/mirror" element={<MirrorPage />} />
      </Routes>
    </Router>
  );
}

export default App;
