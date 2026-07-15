import { LaunchPage } from "@/pages/Launch";
import { LoginPage } from "@/pages/Login";
import { MirrorPage } from "@/pages/Mirror";
import { WebServerClient } from "@/services/webServer";
import { useEffect } from "react";
import { Route, BrowserRouter as Router, Routes } from "react-router-dom";

function AuthExpiryRedirect() {
  useEffect(
    () =>
      WebServerClient.onAuthExpired(() => {
        window.location.replace("/login");
      }),
    []
  );
  return null;
}

function App() {
  return (
    <Router>
      <AuthExpiryRedirect />
      <Routes>
        <Route path="/" element={<LaunchPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/mirror" element={<MirrorPage />} />
      </Routes>
    </Router>
  );
}

export default App;
