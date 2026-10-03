import "./DetectionList.css";

export default function DetectionList({ detections, selectedIndex, onSelect, onConfirm, isSearching }) {
  if (detections.length === 0) {
    return (
      <div className="detection-list detection-list--empty">
        <p>На изображении не найдено ни одного транспортного средства.</p>
        <p className="detection-list__hint">Попробуйте другое фото.</p>
      </div>
    );
  }

  return (
    <div className="detection-list">
      <h3>Найдено ТС: {detections.length}</h3>
      <p className="detection-list__hint">Выберите нужное транспортное средство</p>

      <ul>
        {detections.map((det) => (
          <li key={det.index}>
            <button
              type="button"
              className={`detection-item${det.index === selectedIndex ? " detection-item--selected" : ""}`}
              onClick={() => onSelect(det.index)}
            >
              <span className="detection-item__badge">{det.index}</span>
              <span className="detection-item__label">Транспортное средство</span>
              <span className="num detection-item__confidence">
                {Math.round(det.confidence * 100)}%
              </span>
            </button>
          </li>
        ))}
      </ul>

      <button
        type="button"
        className="btn btn-primary detection-list__submit"
        disabled={selectedIndex === null || isSearching}
        onClick={onConfirm}
      >
        {isSearching ? "Поиск..." : "Найти похожие"}
      </button>
    </div>
  );
}