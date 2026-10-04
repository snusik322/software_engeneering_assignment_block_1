import csv
from pathlib import Path

import librosa
import tensorflow_hub as hub

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000  # YAMNet принимает 16 кГц, моно

_model = None
_names = None


def load_model():
    """Загружает YAMNet и список названий классов один раз."""
    global _model, _names
    if _model is None:
        _model = hub.load(YAMNET_URL)
        path = _model.class_map_path().numpy().decode("utf-8")
        with open(path, encoding="utf-8") as f:
            _names = [row["display_name"] for row in csv.DictReader(f)]
    return _model, _names


def predict(audio_path: str, top_k: int = 5) -> dict:
    """Возвращает top-k классов со средними вероятностями по всей записи."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    model, names = load_model()
    waveform, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError("Пустая аудиозапись")

    scores, _, _ = model(waveform)          # (окна, 521)
    mean_scores = scores.numpy().mean(axis=0)

    order = mean_scores.argsort()[::-1][:top_k]
    top = [{"label": names[i], "score": round(float(mean_scores[i]), 4)} for i in order]
    return {"label": top[0]["label"], "score": top[0]["score"], "top_k": top}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python yamnet_classify.py путь_к_аудио")
        sys.exit(1)
    for item in predict(sys.argv[1])["top_k"]:
        print(f"  {item['label']:<30} {item['score']:.3f}")
