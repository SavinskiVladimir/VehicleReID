import "./ErrorBanner.css";

export default function ErrorBanner({ message, onDismiss }) {
  if (!message) return null;

  return (
    <div className="error-banner" role="alert">
      <span>{message}</span>
      <button type="button" className="error-banner__close" onClick={onDismiss} aria-label="Закрыть">
        ×
      </button>
    </div>
  );
}