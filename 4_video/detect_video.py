from pathlib import Path

import cv2
from PIL import Image
from transformers import pipeline

MODEL_NAME = "facebook/detr-resnet-50"

_detector = None


def load_model():
    global _detector
    if _detector is None:
        _detector = pipeline("object-detection", model=MODEL_NAME)
    return _detector


def detect_video(video_path: str, every: int = 10, threshold: float = 0.8, out_path: str | None = None) -> dict:
    """Детектирует объекты на каждом `every`-м кадре.
    Если задан out_path, сохраняет копию видео с рамками."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    detector = load_model()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames, current, n = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every == 0:
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current = [d for d in detector(rgb) if d["score"] >= threshold]
            frames.append({
                "frame": n,
                "detections": [{"label": d["label"], "score": round(d["score"], 4)} for d in current],
            })
        if writer is not None:
            for d in current:
                b = d["box"]
                p1, p2 = (b["xmin"], b["ymin"]), (b["xmax"], b["ymax"])
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f"{d['label']} {d['score']:.2f}", (p1[0], max(p1[1] - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.write(frame)
        n += 1

    cap.release()
    if writer is not None:
        writer.release()

    return {"file": path.name, "fps": round(fps, 2), "total_frames": total,
            "processed_frames": len(frames), "every": every, "frames": frames}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Детекция объектов на видео")
    p.add_argument("video")
    p.add_argument("--out", default=None, help="Куда сохранить видео с рамками")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.8)
    a = p.parse_args()

    res = detect_video(a.video, a.every, a.threshold, a.out)
    print(f"{res['file']}: {res['fps']} к/с, всего кадров {res['total_frames']}, обработано {res['processed_frames']}")
    for item in res["frames"][:10]:
        labels = ", ".join(f"{d['label']} {d['score']}" for d in item["detections"])
        print(f"  кадр {item['frame']:>4}: {labels or '-'}")
