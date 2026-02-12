import { isTauri } from "@/services/runtime";
import { WebServerClient } from "@/services/webServer";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import LaunchDesktop from "@/pages/LaunchDesktop";

export function LaunchPage() {
  const navigate = useNavigate();
  const isDesktop = isTauri();

  useEffect(() => {
    if (isDesktop) {
      return;
    }
    if (WebServerClient.isAuthed) {
      navigate("/mirror", { replace: true });
    } else {
      navigate("/login", { replace: true });
    }
  }, [isDesktop, navigate]);

  if (isDesktop) {
    return <LaunchDesktop />;
  }

  return null;
}
