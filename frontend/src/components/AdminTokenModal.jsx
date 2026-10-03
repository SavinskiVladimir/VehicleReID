import { useState } from "react";
import "./AdminTokenModal.css";

export default function AdminTokenModal({ onSubmit, onCancel }) {
  const [value, setValue] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    if (value.trim()) onSubmit(value.trim());
  }

  return (
    <div className="admin-modal__backdrop" onClick={onCancel}>
      <form className="admin-modal" onClick={(e) => e.stopPropagation()} onSubmit={handleSubmit}>
        <h3>Токен администратора</h3>
        <p className="admin-modal__hint">
          Это административное действие (сохранение или удаление записи базы).
          Введите токен, чтобы продолжить.
        </p>
        <input
          type="password"
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="X-Admin-Token"
        />
        <div className="admin-modal__actions">
          <button type="button" className="btn btn-secondary" onClick={onCancel}>
            Отмена
          </button>
          <button type="submit" className="btn btn-primary" disabled={!value.trim()}>
            Подтвердить
          </button>
        </div>
      </form>
    </div>
  );
}