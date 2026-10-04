import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
MAX_LENGTH = 512

_tokenizer = None
_model = None


def load_model():
    """Загружает модель один раз и дальше возвращает уже загруженную."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def predict(text: str) -> dict:
    """Возвращает метку, уверенность и вероятности всех классов."""
    if not text or not text.strip():
        raise ValueError("Текст не может быть пустым")

    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]
    best = int(probs.argmax())
    names = model.config.id2label

    return {
        "label": names[best].lower(),
        "score": round(float(probs[best]), 4),
        "probabilities": {names[i].lower(): round(float(p), 4) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    import sys

    texts = sys.argv[1:] or [
        "Фильм посмотрел на одном дыхании, обязательно пересмотрю",
        "Сеанс начинается в девять вечера",
        "Скучно и затянуто, зря потратил время",
    ]
    for t in texts:
        r = predict(t)
        print(f"{r['label']:<9} {r['score']:.3f}  |  {t}")
