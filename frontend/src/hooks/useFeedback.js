import { useCallback, useEffect, useState } from "react";
import { getFeedback, setFeedback, removeFeedback, extractErrorMessage } from "../api/client";
import { useAdminToken } from "./useAdminToken";

/** Отметки оператора по одному запросу. Обновление мгновенное (оптимистичное), при ошибке откатывается. */
export function useFeedback(queryId) {
  const [marks, setMarks] = useState({}); // { candidateId: "match" | "no_match" }
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(null); // отметка, ждущая ввода токена
  const { token, setToken, clearToken } = useAdminToken();

  useEffect(() => {
    setMarks({});
    setError(null);
    if (!queryId) return undefined;
    let cancelled = false;
    getFeedback(queryId)
      .then((items) => {
        if (!cancelled) setMarks(Object.fromEntries(items.map((i) => [i.candidate_id, i.label])));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [queryId]);

  const apply = useCallback(
    async (candidateId, label, tokenValue) => {
      const previous = marks[candidateId] ?? null;
      const next = previous === label ? null : label; // повторное нажатие снимает отметку
      const write = (value) =>
        setMarks((m) => {
          const copy = { ...m };
          if (value) copy[candidateId] = value;
          else delete copy[candidateId];
          return copy;
        });
      write(next);
      try {
        if (next) await setFeedback(queryId, candidateId, next, tokenValue);
        else await removeFeedback(queryId, candidateId, tokenValue);
        setToken(tokenValue);
      } catch (err) {
        write(previous);
        if (err.response?.status === 403) {
          clearToken();
          setError("Неверный токен. Нажмите ещё раз и введите его заново.");
        } else {
          setError(extractErrorMessage(err));
        }
      }
    },
    [marks, queryId, setToken, clearToken]
  );

  const mark = useCallback(
    (candidateId, label) => {
      if (token) apply(candidateId, label, token);
      else setPending({ candidateId, label });
    },
    [token, apply]
  );

  return {
    marks,
    error,
    clearError: () => setError(null),
    mark,
    needsToken: pending !== null,
    submitToken: (value) => {
      const p = pending;
      setPending(null);
      if (p) apply(p.candidateId, p.label, value);
    },
    cancelToken: () => setPending(null),
  };
}