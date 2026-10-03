import { useCallback, useState } from "react";

const STORAGE_KEY = "vehicle-reid-admin-token";

/**
 * Хранит admin-токен (см. backend/app/auth.py) в localStorage, чтобы не
 * запрашивать его повторно при каждом admin-действии в рамках сессии
 * браузера. Токен вводится один раз через AdminTokenModal.
 */
export function useAdminToken() {
  const [token, setTokenState] = useState(() => localStorage.getItem(STORAGE_KEY) || "");

  const setToken = useCallback((value) => {
    localStorage.setItem(STORAGE_KEY, value);
    setTokenState(value);
  }, []);

  const clearToken = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setTokenState("");
  }, []);

  return { token, setToken, clearToken, hasToken: Boolean(token) };
}