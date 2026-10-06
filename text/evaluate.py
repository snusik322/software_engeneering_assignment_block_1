import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sentiment import load_model, predict

DATA = Path(__file__).parent / "data" / "test_samples.json"
LABELS = ["negative", "neutral", "positive"]


def main():
    samples = json.loads(DATA.read_text(encoding="utf-8"))

    t = time.perf_counter()
    load_model()
    load_time = time.perf_counter() - t

    y_true, y_pred, wrong = [], [], []
    t = time.perf_counter()
    for s in samples:
        pred = predict(s["text"])["label"]
        y_true.append(s["label"])
        y_pred.append(pred)
        if pred != s["label"]:
            wrong.append((s["label"], pred, s["text"]))
    run_time = time.perf_counter() - t

    print(f"Загрузка модели: {load_time:.2f} с")
    print(f"Предсказание {len(samples)} текстов: {run_time:.2f} с")
    print(f"В среднем на текст: {run_time / len(samples) * 1000:.1f} мс\n")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))

    print("Матрица ошибок (строки - правда, столбцы - предсказание):")
    print(f"{'':>10}" + "".join(f"{l:>10}" for l in LABELS))
    for l, row in zip(LABELS, confusion_matrix(y_true, y_pred, labels=LABELS)):
        print(f"{l:>10}" + "".join(f"{v:>10}" for v in row))

    print("\nОшибки:")
    for true, pred, text in wrong:
        print(f"  [{true} -> {pred}] {text}")


if __name__ == "__main__":
    main()
