import { isTauri } from "@/services/runtime";
import { WebServerClient } from "@/services/webServer";
import { useEffect, useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import "@/styles/login.css";

export function LoginPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [password, setPassword] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isTauri()) {
      navigate("/");
      return;
    }
    if (WebServerClient.isAuthed) {
      navigate("/mirror");
    }
  }, [navigate]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (loading) {
      return;
    }
    if (!password.trim()) {
      setNotice(t("Please enter password."));
      return;
    }
    setLoading(true);
    const token = await WebServerClient.login(password.trim());
    setLoading(false);
    if (!token) {
      setNotice(t("Invalid password."));
      return;
    }
    setNotice(null);
    navigate("/mirror");
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <h1 className="login-title">{t("Login")}</h1>
        <p className="login-hint">{t("Default password is 123456.")}</p>
        <form className="login-form" onSubmit={handleSubmit}>
          <label className="login-label" htmlFor="login-password">
            {t("Password")}
          </label>
          <input
            id="login-password"
            className="login-input"
            type="password"
            value={password}
            placeholder={t("Enter password")}
            onChange={(event) => {
              setPassword(event.target.value);
              if (notice) {
                setNotice(null);
              }
            }}
            autoFocus
          />
          {notice && <div className="login-notice">{notice}</div>}
          <button
            className={`login-button ${loading ? "disabled" : ""}`}
            type="submit"
            disabled={loading}
          >
            {loading ? t("Logging in...") : t("Sign In")}
          </button>
        </form>
      </div>
    </div>
  );
}