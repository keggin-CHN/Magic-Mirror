import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import LocalStorageBackend from "i18next-localstorage-backend";

import enTranslation from "@/assets/locales/en.json";
import zhTranslation from "@/assets/locales/zh.json";

i18n
  .use(LanguageDetector)
  .use(LocalStorageBackend)
  .use(initReactI18next)
  .init({
    fallbackLng: "en",
    debug: true,
    resources: {
      en: { translation: enTranslation },
      zh: { translation: zhTranslation },
    },
    detection: {
      order: ["localStorage", "navigator"],
      caches: ["localStorage"],
    },
  });

export default i18n;
