import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

import "@/services/i18n";
import "@/styles/global.css";
import "virtual:uno.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
