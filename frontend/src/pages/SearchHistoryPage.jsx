import { useEffect, useState } from "react";
import SearchHistoryList from "../components/SearchHistoryList";
import ResultsGrid from "../components/ResultsGrid";
import ErrorBanner from "../components/ErrorBanner";
import AdminTokenModal from "../components/AdminTokenModal";
import {
  getSearchHistory,
  rerunSearch,
  deleteSearchQuery,
  clearSearchHistory,
  extractErrorMessage,
} from "../api/client";
import { useAdminToken } from "../hooks/useAdminToken";
import { queryCropUrl } from "../utils/files";
import "./SearchHistoryPage.css";

const PAGE_SIZE = 8;
const WRONG_TOKEN_MESSAGE = "Неверный токен администратора. Нажмите ещё раз и введите его заново.";

/**
 * "История поиска" (функция 9 ТЗ): журнал выполненных поисков с повторным
 * поиском. Это НЕ база наблюдений (вкладка "База наблюдений") - там записи,
 * в которых ищем, а здесь - то, что искали.
 */
export default function SearchHistoryPage() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const [rerunningId, setRerunningId] = useState(null);
  const [rerun, setRerun] = useState(null); // { item, response } - открыт повторный поиск
  const [deletingId, setDeletingId] = useState(null);
  const [pendingAction, setPendingAction] = useState(null); // { type: "delete", id } | { type: "clear" }

  const { token, setToken, clearToken } = useAdminToken();

  async function load() {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getSearchHistory(PAGE_SIZE, offset);
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

  async function handleRerun(item) {
    setRerunningId(item.id);
    setError(null);
    try {
      const response = await rerunSearch(item.id);
      setRerun({ item, response });
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setRerunningId(null);
    }
  }

  function requestAdminAction(action) {
    if (token) {
      performAdminAction(action, token);
    } else {
      setPendingAction(action);
    }
  }

  async function performAdminAction(action, adminToken) {
    setError(null);
    if (action.type === "delete") setDeletingId(action.id);
    try {
      if (action.type === "delete") {
        await deleteSearchQuery(action.id, adminToken);
      } else {
        await clearSearchHistory(adminToken);
      }
      setToken(adminToken);
      // После удаления страница могла опустеть - возвращаемся на предыдущую.
      if (action.type === "clear" || (items.length === 1 && offset > 0)) {
        setOffset((o) => (action.type === "clear" ? 0 : Math.max(0, o - PAGE_SIZE)));
        if (action.type === "clear" && offset === 0) load();
      } else {
        load();
      }
    } catch (err) {
      if (err.response?.status === 403) {
        clearToken();
        setError(WRONG_TOKEN_MESSAGE);
      } else {
        setError(extractErrorMessage(err));
      }
    } finally {
      setDeletingId(null);
    }
  }

  // --- Детальный вид: результаты повторного поиска ---
  if (rerun) {
    return (
      <div className="search-history-page">
        <div className="search-history-page__header">
          <h2>Повторный поиск</h2>
          <button type="button" className="btn btn-secondary" onClick={() => setRerun(null)}>
            Назад к истории
          </button>
        </div>

        <div className="search-history-page__rerun-query">
          <img src={queryCropUrl(rerun.item.id)} alt="Искомое транспортное средство" />
          <div>
            <p className="search-history-page__rerun-date">
              Исходный поиск:{" "}
              {new Date(rerun.item.created_at).toLocaleString("ru-RU", {
                dateStyle: "medium",
                timeStyle: "short",
              })}
            </p>
            <p className="search-history-page__note">
              Поиск выполнен по сохранённому признаковому описанию, по текущему содержимому
              базы наблюдений, поэтому результат может отличаться от исходного.
            </p>
            <p className="search-history-page__count">
              Найдено кандидатов: {rerun.response.results.length}
            </p>
          </div>
        </div>

        <ResultsGrid results={rerun.response.results} />
      </div>
    );
  }

  const hasPrev = offset > 0;
  const hasNext = offset + PAGE_SIZE < total;

  return (
    <div className="search-history-page">
      <div className="search-history-page__header">
        <div>
          <h2>История поиска</h2>
          <p className="search-history-page__count">Выполнено поисков: {total}</p>
        </div>
        {total > 0 && (
          <button
            type="button"
            className="btn btn-danger"
            onClick={() => requestAdminAction({ type: "clear" })}
          >
            Очистить историю
          </button>
        )}
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {isLoading ? (
        <p className="search-history-page__loading">Загрузка...</p>
      ) : (
        <>
          <SearchHistoryList
            items={items}
            onRerun={handleRerun}
            onDelete={(id) => requestAdminAction({ type: "delete", id })}
            rerunningId={rerunningId}
            deletingId={deletingId}
          />

          {total > PAGE_SIZE && (
            <div className="search-history-page__pager">
              <button
                type="button"
                className="btn btn-secondary"
                disabled={!hasPrev}
                onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
              >
                Назад
              </button>
              <span className="num search-history-page__pager-label">
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

      {pendingAction && (
        <AdminTokenModal
          onSubmit={(value) => {
            const action = pendingAction;
            setPendingAction(null);
            performAdminAction(action, value);
          }}
          onCancel={() => setPendingAction(null)}
        />
      )}
    </div>
  );
}