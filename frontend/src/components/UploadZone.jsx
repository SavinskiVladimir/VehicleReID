import { useCallback, useRef, useState } from "react";
import "./UploadZone.css";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export default function UploadZone({ onFileSelected, isProcessing }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = useCallback(
    (files) => {
      const file = files?.[0];
      if (!file) return;
      if (!ACCEPTED_TYPES.includes(file.type)) return;
      onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div
      className={`upload-zone${isDragging ? " upload-zone--dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />

      <svg className="upload-zone__icon" width="40" height="40" viewBox="0 0 24 24" fill="none">
        <path
          d="M12 16V4M12 4L7 9M12 4l5 5"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>

      <p className="upload-zone__title">
        {isProcessing ? "Обрабатываю изображение..." : "Перетащите кадр сюда или нажмите, чтобы выбрать файл"}
      </p>
      <p className="upload-zone__hint">JPEG, PNG или WebP, до 10 МБ</p>
    </div>
  );
}