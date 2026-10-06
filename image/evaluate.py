import time
from pathlib import Path

from classify import categories, load_model, predict

DATA = Path(__file__).parent / "data"
EXT = {".jpg", ".jpeg", ".png", ".webp"}


GROUP_RANGES = {
    "cat": [(281, 285)],   
    "dog": [(151, 268)],   
}

GROUP_NAMES = {
    "car": ["sports car", "convertible", "cab", "minivan", "limousine", "pickup",
            "racer", "beach wagon", "jeep", "Model T"],
}


def accepted_labels(folder_name, names):
    """Какие классы ImageNet считаем верным ответом для этой папки."""
    key = folder_name.lower().replace("_", " ")
    if key in GROUP_RANGES:
        return {names[i] for a, b in GROUP_RANGES[key] for i in range(a, b + 1)}
    if key in GROUP_NAMES:
        return {n for n in names if n in GROUP_NAMES[key]}
    exact = {n for n in names if n.lower() == key}
    if exact:
        return exact
    return {n for n in names if key in n.lower()}   # запасной вариант: название содержит слово


def main():
    names = categories()
    items = []
    for folder in sorted(p for p in DATA.iterdir() if p.is_dir()):
        accepted = accepted_labels(folder.name, names)
        if not accepted:
            print(f"Пропуск: для папки «{folder.name}» не нашлось подходящих классов ImageNet")
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in EXT:
                items.append((f, folder.name, accepted))

    if not items:
        print("Изображения не найдены. Положите фото в data/<папка>/")
        return

    t = time.perf_counter()
    load_model()
    load_time = time.perf_counter() - t

    top1 = top5 = 0
    errors = []
    t = time.perf_counter()
    for f, folder, accepted in items:
        res = predict(str(f), top_k=5)
        top1 += res["label"] in accepted
        top5 += any(x["label"] in accepted for x in res["top_k"])
        if res["label"] not in accepted:
            errors.append((f"{folder}/{f.name}", res["label"], res["score"]))
    run_time = time.perf_counter() - t

    n = len(items)
    print(f"\nЗагрузка модели: {load_time:.2f} с")
    print(f"Классификация {n} изображений: {run_time:.2f} с (в среднем {run_time / n * 1000:.1f} мс)\n")
    print(f"Top-1 accuracy: {top1 / n:.3f}  ({top1} из {n})")
    print(f"Top-5 accuracy: {top5 / n:.3f}  ({top5} из {n})\n")
    if errors:
        print("Ошибки top-1:")
        for name, got, score in errors:
            print(f"  {name:<30} модель ответила: {got} ({score})")


if __name__ == "__main__":
    main()