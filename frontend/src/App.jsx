import { useState } from "react";
import Header from "./components/Header";
import SearchPage from "./pages/SearchPage";
import SearchHistoryPage from "./pages/SearchHistoryPage";
import ObservationsPage from "./pages/ObservationsPage";
import "./App.css";

const PAGES = {
  search: SearchPage,
  history: SearchHistoryPage, // журнал выполненных поисков
  observations: ObservationsPage, // база наблюдений - то, в чём ищем
};

export default function App() {
  const [page, setPage] = useState("search");
  const Page = PAGES[page];

  return (
    <>
      <Header page={page} onNavigate={setPage} />
      <main className="app-main">
        <Page />
      </main>
    </>
  );
}