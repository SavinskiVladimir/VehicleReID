import { useEffect, useRef, useState } from "react";
import UploadZone from "../components/UploadZone";
import DetectionCanvas from "../components/DetectionCanvas";
import DetectionList from "../components/DetectionList";
import ResultsGrid from "../components/ResultsGrid";
import ErrorBanner from "../components/ErrorBanner";
import AdminTokenModal from "../components/AdminTokenModal";
import { detectVehicles, searchVehicle, addObservation, extractErrorMessage } from "../api/client";
import { useAdminToken } from "../hooks/useAdminToken";
import "./SearchPage.css";

// "idle" -> "detecting" -> "detected" -> "searching" -> "results"
export default function SearchPage() {
  const [stage, setStage] = useState("idle");
  const [imageUrl, setImageUrl] = useState(null);
  const [uploadId, setUploadId] = useState(null);
  const [detections, setDetections] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [searchResponse, setSearchResponse] = useState(null);
  const [error, setError] = useState(null);
  const [showAdminModal, setShowAdminModal] = useState(false);
  const [saveStatus, setSaveStatus] = useState("idle"); // idle | saving | saved

  const { token, setToken, clearToken } = useAdminToken();
  const objectUrlRef = useRef(null);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  async function handleFileSelected(file) {
    setError(null);
    setSaveStatus("idle");

    const url = URL.createObjectURL(file);
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = url;
    setImageUrl(url);
    setStage("detecting");

    try {
      const data = await detectVehicles(file);
      setUploadId(data.upload_id);
      setDetections(data.detections);
      setSelectedIndex(data.detections.length === 1 ? 0 : null);
      setStage("detected");
    } catch (err) {
      setError(extractErrorMessage(err));
      setStage("idle");
    }
  }

  async function handleSearch() {
    if (selectedIndex === null) return;
    setError(null);
    setStage("searching");

    try {
      const response = await searchVehicle(uploadId, selectedIndex, 10);
      setSearchResponse(response);
      setStage("results");
    } catch (err) {
      setError(extractErrorMessage(err));
      setStage("detected");
    }
  }

  async function handleSaveObservation(adminToken) {
    setSaveStatus("saving");
    setError(null);
    try {
      await addObservation(uploadId, selectedIndex, adminToken);
      setToken(adminToken);
      setSaveStatus("saved");
    } catch (err) {
      if (err.response?.status === 403) {
        // Токен не подошёл (например, ADMIN_API_KEY в .env поменяли после
        // того, как браузер его запомнил). Забываем запомненное значение -
        // иначе оно отправлялось бы снова и снова, а окно ввода не
        // появлялось бы никогда.
        clearToken();
        setError("Неверный токен администратора. Нажмите ещё раз и введите его заново.");
      } else {
        setError(extractErrorMessage(err));
      }
      setSaveStatus("idle");
    }
  }

  function handleSaveClick() {
    if (token) {
      handleSaveObservation(token);
    } else {
      setShowAdminModal(true);
    }
  }

  function handleReset() {
    setStage("idle");
    setImageUrl(null);
    setUploadId(null);
    setDetections([]);
    setSelectedIndex(null);
    setSearchResponse(null);
    setSaveStatus("idle");
    setError(null);
  }

  return (
    <div className="search-page">
      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {stage === "idle" && (
        <UploadZone onFileSelected={handleFileSelected} isProcessing={false} />
      )}

      {stage === "detecting" && <UploadZone onFileSelected={() => {}} isProcessing />}

      {(stage === "detected" || stage === "searching") && (
        <div className="search-page__workbench">
          <DetectionCanvas
            imageUrl={imageUrl}
            detections={detections}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
          />
          <DetectionList
            detections={detections}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
            onConfirm={handleSearch}
            isSearching={stage === "searching"}
          />
        </div>
      )}

      {stage === "results" && searchResponse && (
        <div className="search-page__results">
          <div className="search-page__results-header">
            <div>
              <h2>Результаты поиска</h2>
              <p className="search-page__results-count">
                Найдено кандидатов: {searchResponse.results.length}
              </p>
            </div>
            <div className="search-page__results-actions">
              <button type="button" className="btn btn-secondary" onClick={handleReset}>
                Новый поиск
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleSaveClick}
                disabled={saveStatus === "saving" || saveStatus === "saved"}
              >
                {saveStatus === "saved"
                  ? "Сохранено"
                  : saveStatus === "saving"
                  ? "Сохранение..."
                  : "Сохранить как наблюдение"}
              </button>
            </div>
          </div>

          <ResultsGrid results={searchResponse.results} />
        </div>
      )}

      {showAdminModal && (
        <AdminTokenModal
          onSubmit={(value) => {
            setShowAdminModal(false);
            handleSaveObservation(value);
          }}
          onCancel={() => setShowAdminModal(false)}
        />
      )}
    </div>
  );
}