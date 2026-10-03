import { useState } from "react";
import Header from "./components/Header";
import SearchPage from "./pages/SearchPage";
import HistoryPage from "./pages/HistoryPage";
import "./App.css";

export default function App() {
  const [page, setPage] = useState("search");

  return (
    <>
      <Header page={page} onNavigate={setPage} />
      <main className="app-main">
        {page === "search" ? <SearchPage /> : <HistoryPage />}
      </main>
    </>
  );
}