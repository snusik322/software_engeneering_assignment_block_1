import time
from collections import Counter
from pathlib import Path

from detect_video import detect_video, load_model

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"
EXT = {".mp4", ".mov", ".avi", ".mkv"}
EVERY = 10
THRESHOLD = 0.8


def main():
    videos = sorted(f for f in DATA.iterdir() if f.suffix.lower() in EXT)
    if not videos:
        print("Видео не найдены. Положите ролики в data/ (cup.mp4, ...)")
        return

    t = time.perf_counter()
    load_model()
    print(f"Загрузка модели: {time.perf_counter() - t:.2f} с\n")

    print(f"{'файл':<16}{'кадров':>7}{'detect':>8}{'correct':>9}{'conf':>8}{'switch':>8}{'stab':>7}")
    totals, all_frames, run = [], 0, 0.0
    wrong = Counter()

    for v in videos:
        expected = v.stem.split("_")[0]
        t = time.perf_counter()
        res = detect_video(str(v), every=EVERY, threshold=THRESHOLD, out_path=str(OUT / f"{v.stem}_detected.mp4"))
        run += time.perf_counter() - t

        fr = res["frames"]
        n = len(fr)
        all_frames += n
        detected = sum(1 for f in fr if f["detections"])
        correct = sum(1 for f in fr if any(d["label"] == expected for d in f["detections"]))
        confs = [d["score"] for f in fr for d in f["detections"] if d["label"] == expected]
        dominant = [max(f["detections"], key=lambda d: d["score"])["label"] for f in fr if f["detections"]]
        switches = sum(1 for a, b in zip(dominant, dominant[1:]) if a != b)
        stability = 1 - switches / max(len(dominant) - 1, 1)
        for d in dominant:
            if d != expected:
                wrong[(expected, d)] += 1

        mean_conf = sum(confs) / len(confs) if confs else 0.0
        totals.append((correct / n if n else 0, stability))
        print(f"{v.name:<16}{n:>7}{detected / n:>8.3f}{correct / n:>9.3f}{mean_conf:>8.3f}{switches:>8}{stability:>7.3f}")

    print(f"\nСредняя доля кадров с верным классом: {sum(c for c, _ in totals) / len(totals):.3f}")
    print(f"Средняя стабильность: {sum(s for _, s in totals) / len(totals):.3f}")
    print(f"Обработано {all_frames} кадров за {run:.1f} с ({run / max(all_frames, 1) * 1000:.0f} мс на кадр)")
    if wrong:
        print("\nНаиболее частые «чужие» доминирующие классы (ожидался -> получен):")
        for (exp, got), c in wrong.most_common(5):
            print(f"  {exp} -> {got}: {c} кадров")


if __name__ == "__main__":
    main()
