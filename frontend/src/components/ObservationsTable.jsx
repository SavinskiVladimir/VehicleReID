import { cropUrl } from "../utils/files";
import "./ObservationsTable.css";

export default function ObservationsTable({ items, onDelete, deletingId }) {
  if (items.length === 0) {
    return (
      <div className="observations-table observations-table--empty">
        <p>База наблюдений пуста.</p>
        <p className="observations-table__hint">Добавьте записи кнопкой «Сохранить как наблюдение» после поиска.</p>
      </div>
    );
  }

  return (
    <table className="observations-table">
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
              <img className="observations-table__thumb" src={cropUrl(obs.id)} alt="" loading="lazy" />
            </td>
            <td>
              {new Date(obs.created_at).toLocaleString("ru-RU", { dateStyle: "medium", timeStyle: "short" })}
            </td>
            <td className="num observations-table__bbox">
              {obs.bbox.x1}, {obs.bbox.y1} – {obs.bbox.x2}, {obs.bbox.y2}
            </td>
            <td>
              <button
                type="button"
                className="btn btn-danger observations-table__delete"
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