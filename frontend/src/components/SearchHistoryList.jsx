import SimilarityBar from "./SimilarityBar";
import { queryCropUrl } from "../utils/files";
import "./SearchHistoryList.css";

/**
 * Журнал выполненных поисков. Намеренно НЕ та же таблица, что у базы
 * наблюдений: там - записи, В КОТОРЫХ ищем (изображение, координаты,
 * удаление); здесь - то, ЧТО искали, с итогом и действием "повторить".
 */
export default function SearchHistoryList({ items, onRerun, onDelete, rerunningId, deletingId }) {
  if (items.length === 0) {
    return (
      <div className="search-history-list search-history-list--empty">
        <p>Поисков пока не было.</p>
        <p className="search-history-list__hint">
          Выполненные поиски сохраняются здесь автоматически и их можно повторить.
        </p>
      </div>
    );
  }

  return (
    <ul className="search-history-list">
      {items.map((item) => (
        <li className="search-history-item" key={item.id}>
          <img
            className="search-history-item__thumb"
            src={queryCropUrl(item.id)}
            alt="Искомое транспортное средство"
            loading="lazy"
          />

          <div className="search-history-item__when">
            <time>
              {new Date(item.created_at).toLocaleString("ru-RU", { dateStyle: "medium", timeStyle: "short" })}
            </time>
            <span className="search-history-item__found">
              {item.result_count === 0
                ? "Ничего не найдено"
                : `Найдено кандидатов: ${item.result_count}`}
            </span>
          </div>

          <div className="search-history-item__best">
            {item.top_similarity !== null && (
              <>
                <span className="search-history-item__best-label">Лучшее совпадение</span>
                <SimilarityBar value={item.top_similarity} />
              </>
            )}
          </div>

          <div className="search-history-item__actions">
            <button
              type="button"
              className="btn btn-primary search-history-item__btn"
              onClick={() => onRerun(item)}
              disabled={rerunningId === item.id}
            >
              {rerunningId === item.id ? "Поиск..." : "Повторить поиск"}
            </button>
            <button
              type="button"
              className="btn btn-danger search-history-item__btn"
              onClick={() => onDelete(item.id)}
              disabled={deletingId === item.id}
            >
              {deletingId === item.id ? "Удаление..." : "Удалить"}
            </button>
          </div>
        </li>
      ))}
    </ul>
  );
}