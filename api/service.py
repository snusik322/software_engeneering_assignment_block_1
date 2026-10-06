from text import sentiment

MODEL_NAME = sentiment.MODEL_NAME

_ready = False


def warm_up() -> None:
    """Загружает модель заранее, при старте приложения."""
    global _ready
    sentiment.load_model()
    _ready = True


def is_ready() -> bool:
    return _ready


def analyze(text: str) -> dict:
    """Определяет тональность одного текста."""
    return sentiment.predict(text)
