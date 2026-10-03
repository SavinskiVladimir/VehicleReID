import "./Header.css";

export default function Header({ page, onNavigate }) {
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <rect x="3" y="10" width="18" height="8" rx="2" stroke="currentColor" strokeWidth="1.8" />
          <circle cx="7.5" cy="18" r="1.6" fill="currentColor" />
          <circle cx="16.5" cy="18" r="1.6" fill="currentColor" />
          <path d="M5 10l1.5-4h11L19 10" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
        </svg>
        <span>Vehicle Re-ID</span>
      </div>

      <nav className="app-header__nav">
        <button
          type="button"
          className={page === "search" ? "is-active" : ""}
          onClick={() => onNavigate("search")}
        >
          Поиск
        </button>
        <button
          type="button"
          className={page === "history" ? "is-active" : ""}
          onClick={() => onNavigate("history")}
        >
          История
        </button>
      </nav>
    </header>
  );
}