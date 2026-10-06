from pathlib import Path

import torch
from PIL import Image
from torchvision import models

WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V2

_model = None
_preprocess = None


def load_model():
    """Создаёт модель и преобразования один раз."""
    global _model, _preprocess
    if _model is None:
        _model = models.resnet50(weights=WEIGHTS).eval()
        _preprocess = WEIGHTS.transforms()
    return _model, _preprocess


def categories():
    return WEIGHTS.meta["categories"]


def predict(image_path: str, top_k: int = 5) -> dict:
    """Возвращает лучший класс и top-k вариантов с вероятностями."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    model, preprocess = load_model()
    img = Image.open(path).convert("RGB")
    batch = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        probs = model(batch).softmax(dim=1)[0]

    values, indices = probs.topk(top_k)
    names = categories()
    top = [{"label": names[int(i)], "score": round(float(v), 4)} for v, i in zip(values, indices)]
    return {"label": top[0]["label"], "score": top[0]["score"], "top_k": top}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python classify.py путь_к_изображению")
        sys.exit(1)
    for item in predict(sys.argv[1])["top_k"]:
        print(f"  {item['label']:<30} {item['score']:.3f}")
