import { useEffect, useState } from "react";
import HistoryTable from "../components/HistoryTable";
import ErrorBanner from "../components/ErrorBanner";
import AdminTokenModal from "../components/AdminTokenModal";
import { getHistory, deleteObservation, extractErrorMessage } from "../api/client";
import { useAdminToken } from "../hooks/useAdminToken";
import "./HistoryPage.css";

const PAGE_SIZE = 12;

export default function HistoryPage() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [pendingDeleteId, setPendingDeleteId] = useState(null);

  const { token, setToken, clearToken } = useAdminToken();

  async function load() {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getHistory(PAGE_SIZE, offset);
      setItems(data.items);
      setTotal(data.total);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offset]);

  function requestDelete(id) {
    if (token) {
      performDelete(id, token);
    } else {
      setPendingDeleteId(id);
    }
  }

  async function performDelete(id, adminToken) {
    setDeletingId(id);
    setError(null);
    try {
      await deleteObservation(id, adminToken);
      setToken(adminToken);
      setItems((prev) => prev.filter((obs) => obs.id !== id));
      setTotal((prev) => prev - 1);
    } catch (err) {
      if (err.response?.status === 403) {
        // См. комментарий в SearchPage.jsx: неверный токен не должен
        // оставаться запомненным, иначе окно ввода больше не появится.
        clearToken();
        setError("Неверный токен администратора. Нажмите ещё раз и введите его заново.");
      } else {
        setError(extractErrorMessage(err));
      }
    } finally {
      setDeletingId(null);
    }
  }

  const hasPrev = offset > 0;
  const hasNext = offset + PAGE_SIZE < total;

  return (
    <div className="history-page">
      <div className="history-page__header">
        <h2>История наблюдений</h2>
        <p className="history-page__count">Всего записей: {total}</p>
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {isLoading ? (
        <p className="history-page__loading">Загрузка...</p>
      ) : (
        <>
          <HistoryTable items={items} onDelete={requestDelete} deletingId={deletingId} />

          {total > PAGE_SIZE && (
            <div className="history-page__pager">
              <button
                type="button"
                className="btn btn-secondary"
                disabled={!hasPrev}
                onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
              >
                Назад
              </button>
              <span className="num history-page__pager-label">
                {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} из {total}
              </span>
              <button
                type="button"
                className="btn btn-secondary"
                disabled={!hasNext}
                onClick={() => setOffset((o) => o + PAGE_SIZE)}
              >
                Далее
              </button>
            </div>
          )}
        </>
      )}

      {pendingDeleteId && (
        <AdminTokenModal
          onSubmit={(value) => {
            const id = pendingDeleteId;
            setPendingDeleteId(null);
            performDelete(id, value);
          }}
          onCancel={() => setPendingDeleteId(null)}
        />
      )}
    </div>
  );
}