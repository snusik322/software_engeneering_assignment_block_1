import time
from pathlib import Path

from yamnet_classify import load_model, predict

DATA = Path(__file__).parent / "data"
EXT = {".wav", ".mp3", ".flac", ".ogg"}

FOLDER_TO_LABEL = {
    "dog": "Dog",
    "cat": "Cat",
    "speech": "Speech",
    "music": "Music",
    "siren": "Siren",
    "rain": "Rain",
    "knock": "Knock",
    "bird": "Bird",
}


def main():
    t = time.perf_counter()
    _, names = load_model()
    load_time = time.perf_counter() - t

    items = []
    for folder, label in FOLDER_TO_LABEL.items():
        d = DATA / folder
        if not d.exists():
            continue
        if label not in names:
            print(f"Пропуск: «{label}» нет среди классов YAMNet")
            continue
        items += [(f, label) for f in sorted(d.iterdir()) if f.suffix.lower() in EXT]

    if not items:
        print("Аудио не найдено. Положите записи в data/<класс>/")
        return

    top1 = top5 = 0
    errors = []
    t = time.perf_counter()
    for f, label in items:
        res = predict(str(f), top_k=5)
        top1 += res["label"] == label
        top5 += label in [x["label"] for x in res["top_k"]]
        if res["label"] != label:
            shown = ", ".join(f"{x['label']} {x['score']:.2f}" for x in res["top_k"][:3])
            errors.append((f"{f.parent.name}/{f.name}", label, shown))
    run_time = time.perf_counter() - t

    n = len(items)
    print(f"Загрузка модели: {load_time:.2f} с")
    print(f"Классификация {n} записей: {run_time:.2f} с (в среднем {run_time / n * 1000:.0f} мс)\n")
    print(f"Top-1 accuracy: {top1 / n:.3f}  ({top1} из {n})")
    print(f"Top-5 accuracy: {top5 / n:.3f}  ({top5} из {n})\n")
    if errors:
        print("Ошибки top-1:")
        for name, true, shown in errors:
            print(f"  {name:<28} ожидалось: {true:<8} получено: {shown}")


if __name__ == "__main__":
    main()
