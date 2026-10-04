import SimilarityBar from "./SimilarityBar";
import { cropUrl } from "../utils/files";
import "./ResultsGrid.css";

export default function ResultsGrid({ results }) {
  if (results.length === 0) {
    return (
      <div className="results-grid results-grid--empty">
        <p>В базе не нашлось ни одного похожего наблюдения.</p>
      </div>
    );
  }

  return (
    <div className="results-grid">
      {results.map((item, i) => (
        <article className="result-card" key={item.observation.id}>
          <div className="result-card__rank">{i + 1}</div>
          <img
            className="result-card__image"
            src={cropUrl(item.observation.id)}
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
          </div>
        </article>
      ))}
    </div>
  );
}