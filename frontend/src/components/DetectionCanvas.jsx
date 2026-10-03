import { useRef, useState } from "react";
import "./DetectionCanvas.css";

/**
 * Изображение с рамками детекций поверх него (SVG, координаты - в
 * исходном пиксельном пространстве кадра). Клик по рамке или её
 * номеру выбирает конкретное ТС для поиска (требование ТЗ: "при
 * наличии нескольких ТС система должна предоставлять пользователю
 * возможность выбора").
 */
export default function DetectionCanvas({ imageUrl, detections, selectedIndex, onSelect }) {
  const imgRef = useRef(null);
  const [naturalSize, setNaturalSize] = useState(null);

  function handleLoad() {
    const img = imgRef.current;
    if (img) {
      setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    }
  }

  return (
    <div
      className="detection-canvas"
      style={naturalSize ? { aspectRatio: `${naturalSize.width} / ${naturalSize.height}` } : undefined}
    >
      <img ref={imgRef} src={imageUrl} alt="Загруженный кадр" onLoad={handleLoad} />

      {naturalSize && detections.length > 0 && (
        <svg
          className="detection-canvas__overlay"
          viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
          preserveAspectRatio="none"
        >
          {detections.map((det) => {
            const { x1, y1, x2, y2 } = det.bbox;
            const isSelected = det.index === selectedIndex;
            return (
              <g
                key={det.index}
                className={`detection-box${isSelected ? " detection-box--selected" : ""}`}
                onClick={() => onSelect(det.index)}
                role="button"
                tabIndex={0}
                aria-label={`Транспортное средство ${det.index}, достоверность ${Math.round(det.confidence * 100)}%`}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") onSelect(det.index);
                }}
              >
                <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} vectorEffect="non-scaling-stroke" />
                <rect
                  className="detection-box__label-bg"
                  x={x1}
                  y={Math.max(0, y1 - 22)}
                  width={28}
                  height={22}
                />
                <text x={x1 + 8} y={Math.max(0, y1 - 22) + 16} className="detection-box__label-text">
                  {det.index}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}