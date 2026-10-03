import "./HistoryTable.css";

export default function HistoryTable({ items, onDelete, deletingId }) {
  if (items.length === 0) {
    return (
      <div className="history-table history-table--empty">
        <p>Наблюдений пока нет.</p>
        <p className="history-table__hint">Сохранённые записи появятся здесь после поиска.</p>
      </div>
    );
  }

  return (
    <table className="history-table">
      <thead>
        <tr>
          <th></th>
          <th>Добавлено</th>
          <th>Координаты</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {items.map((obs) => (
          <tr key={obs.id}>
            <td>
              <img className="history-table__thumb" src={toFileUrl(obs.crop_path)} alt="" loading="lazy" />
            </td>
            <td>
              {new Date(obs.created_at).toLocaleString("ru-RU", { dateStyle: "medium", timeStyle: "short" })}
            </td>
            <td className="num history-table__bbox">
              {obs.bbox.x1}, {obs.bbox.y1} – {obs.bbox.x2}, {obs.bbox.y2}
            </td>
            <td>
              <button
                type="button"
                className="btn btn-danger history-table__delete"
                onClick={() => onDelete(obs.id)}
                disabled={deletingId === obs.id}
              >
                {deletingId === obs.id ? "Удаление..." : "Удалить"}
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function toFileUrl(path) {
  const filename = path.split("/").pop();
  return `/files/crops/${filename}`;
}