import { useState } from "react";
import "./AdminTokenModal.css";

export default function AdminTokenModal({ onSubmit, onCancel }) {
  const [value, setValue] = useState("");

  // HTTP-заголовки передают только однобайтовые символы: токен с
  // кириллицей браузер отправить не сможет, а сервер его не примет.
  const isAscii = /^[\x20-\x7E]*$/.test(value);

  function handleSubmit(e) {
    e.preventDefault();
    if (value.trim() && isAscii) onSubmit(value.trim());
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
        {!isAscii && (
          <p className="admin-modal__error">
            Токен должен состоять из латинских букв, цифр и знаков - кириллица в заголовке не передаётся.
          </p>
        )}
        <div className="admin-modal__actions">
          <button type="button" className="btn btn-secondary" onClick={onCancel}>
            Отмена
          </button>
          <button type="submit" className="btn btn-primary" disabled={!value.trim() || !isAscii}>
            Подтвердить
          </button>
        </div>
      </form>
    </div>
  );
}