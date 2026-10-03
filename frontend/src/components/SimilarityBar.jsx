import "./SimilarityBar.css";

/** Similarity - одновременно число (точное измерение) и бар (быстрое сравнение на глаз). */
export default function SimilarityBar({ value }) {
  const percent = Math.round(value * 100);

  return (
    <div className="similarity-bar">
      <div className="similarity-bar__track">
        <div className="similarity-bar__fill" style={{ width: `${percent}%` }} />
      </div>
      <span className="num similarity-bar__value">{percent}%</span>
    </div>
  );
}