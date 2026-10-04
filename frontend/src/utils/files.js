/**
 * Картинки наблюдений отдаёт сам backend по id записи
 * (GET /api/observations/{id}/crop и /image): он читает путь к файлу
 * из БД. Фронтенду не нужно знать ни где лежит файл на диске сервера,
 * ни какая там ОС (на Windows пути с \, на Linux - с /), ни попал ли он
 * в базу через API или через демо-скрипт с путём на исходный датасет.
 */
export function cropUrl(observationId) {
  return `/api/observations/${observationId}/crop`;
}

export function imageUrl(observationId) {
  return `/api/observations/${observationId}/image`;
}