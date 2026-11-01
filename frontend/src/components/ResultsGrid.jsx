import SimilarityBar from "./SimilarityBar";
import ErrorBanner from "./ErrorBanner";
import AdminTokenModal from "./AdminTokenModal";
import { useFeedback } from "../hooks/useFeedback";
import { cropUrl } from "../utils/files";
import "./ResultsGrid.css";

export default function ResultsGrid({ results, queryId }) {
  const fb = useFeedback(queryId);

  if (results.length === 0) {
    return (
      <div className="results-grid results-grid--empty">
        <p>В базе не нашлось ни одного похожего наблюдения.</p>
      </div>
    );
  }

  return (
    <>
      <ErrorBanner message={fb.error} onDismiss={fb.clearError} />
      <div className="results-grid">
        {results.map((item, i) => {
          const id = item.observation.id;
          const mark = fb.marks[id];
          return (
            <article className="result-card" key={id}>
              <div className="result-card__rank">{i + 1}</div>
              <img
                className="result-card__image"
                src={cropUrl(id)}
                alt={`Кандидат ${i + 1}`}
                loading="lazy"
              />
              <div className="result-card__body">
                <SimilarityBar value={item.similarity} />
                <time className="result-card__date">
                  {new Date(item.observation.created_at).toLocaleString("ru-RU", {
                    dateStyle: "short",
                    timeStyle: "short",
                  })}
                </time>
                {queryId && (
                  <div className="result-card__feedback">
                    <button
                      type="button"
                      className={`result-card__mark result-card__mark--match${mark === "match" ? " is-active" : ""}`}
                      onClick={() => fb.mark(id, "match")}
                      title="Это то же ТС"
                    >
                      ✓ Совпадает
                    </button>
                    <button
                      type="button"
                      className={`result-card__mark result-card__mark--no_match${mark === "no_match" ? " is-active" : ""}`}
                      onClick={() => fb.mark(id, "no_match")}
                      title="Это другое ТС"
                    >
                      ✕ Нет
                    </button>
                  </div>
                )}
              </div>
            </article>
          );
        })}
      </div>
      {fb.needsToken && <AdminTokenModal onSubmit={fb.submitToken} onCancel={fb.cancelToken} />}
    </>
  );
}