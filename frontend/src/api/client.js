import axios from "axios";

// Путь ОТНОСИТЕЛЬНЫЙ по умолчанию (не http://localhost:8000) - в dev
// Vite проксирует /api на backend (см. vite.config.js), в проде это
// будет делать reverse-proxy перед обоими сервисами (этап 9 ТЗ,
// "Развёртывание"). Благодаря этому фронтенд и backend всегда видятся
// браузеру как один origin - отдельная настройка CORS не нужна вообще.
// VITE_API_BASE_URL - аварийный выход, если понадобится обратиться к
// backend на другом origin напрямую.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

export const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 15000,
});

/**
 * Единообразно превращает ошибку axios в текст, понятный пользователю.
 * Backend (см. routes.py) всегда возвращает { detail: "..." } в теле
 * ошибки - используем это сообщение напрямую, если оно есть.
 */
export function extractErrorMessage(error) {
  if (error.response?.data?.detail) {
    return error.response.data.detail;
  }
  if (error.code === "ECONNABORTED") {
    return "Сервер не ответил вовремя. Попробуйте ещё раз.";
  }
  if (!error.response) {
    return "Не удалось связаться с сервером. Проверьте подключение.";
  }
  return "Произошла непредвиденная ошибка.";
}

// --- Функции 1-2 ТЗ: загрузка изображения + детекция ---
export async function detectVehicles(file) {
  const formData = new FormData();
  formData.append("file", file);
  // ВАЖНО: НЕ указывать Content-Type вручную. Когда тело - FormData,
  // браузер сам генерирует заголовок с правильным boundary, уникальным
  // для каждого запроса. Если задать "multipart/form-data" явно (без
  // boundary), заголовок и реальное тело запроса расходятся - сервер
  // не может разобрать форму ("Missing boundary in multipart.").
  const { data } = await api.post("/detect", formData);
  return data; // { upload_id, detections: [{ index, bbox, confidence }] }
}

// --- Функции 3-7 ТЗ: кроп + эмбеддинг + поиск ---
export async function searchVehicle(uploadId, detectionIndex, topK = 10) {
  const { data } = await api.post("/search", {
    upload_id: uploadId,
    detection_index: detectionIndex,
    top_k: topK,
  });
  return data; // { query_bbox, results: [{ observation, similarity }] }
}

// --- Функция 8 ТЗ: добавление наблюдения (admin) ---
export async function addObservation(uploadId, detectionIndex, adminToken) {
  const { data } = await api.post(
    "/observations",
    { upload_id: uploadId, detection_index: detectionIndex },
    { headers: { "X-Admin-Token": adminToken } }
  );
  return data;
}

// --- База наблюдений: список записей, в которых ищем ---
export async function getObservations(limit = 20, offset = 0) {
  const { data } = await api.get("/observations", { params: { limit, offset } });
  return data; // { items, total, limit, offset }
}

// --- Удаление наблюдения (admin) ---
export async function deleteObservation(observationId, adminToken) {
  await api.delete(`/observations/${observationId}`, {
    headers: { "X-Admin-Token": adminToken },
  });
}

// --- История поиска (функция 9 ТЗ) - журнал выполненных поисков, ОТДЕЛЬНО от базы наблюдений ---
export async function getSearchHistory(limit = 10, offset = 0) {
  const { data } = await api.get("/search-history", { params: { limit, offset } });
  return data; // { items: [{ id, created_at, bbox, top_k, result_count, top_similarity }], total, limit, offset }
}

// Повторный поиск по сохранённому вектору: ни файл, ни токен не нужны.
export async function rerunSearch(queryId) {
  const { data } = await api.post(`/search-history/${queryId}/search`);
  return data; // как у обычного поиска: { query_bbox, results, query_id }
}

export async function deleteSearchQuery(queryId, adminToken) {
  await api.delete(`/search-history/${queryId}`, { headers: { "X-Admin-Token": adminToken } });
}

export async function clearSearchHistory(adminToken) {
  const { data } = await api.delete("/search-history", { headers: { "X-Admin-Token": adminToken } });
  return data; // { deleted: N }
}