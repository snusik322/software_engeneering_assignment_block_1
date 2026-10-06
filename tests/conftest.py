import pytest
from fastapi.testclient import TestClient

from api import service
from api.main import app

CANNED = {
    "positive": {"negative": 0.05, "neutral": 0.15, "positive": 0.80},
    "negative": {"negative": 0.85, "neutral": 0.10, "positive": 0.05},
    "neutral": {"negative": 0.10, "neutral": 0.80, "positive": 0.10},
}


def fake_prediction(text: str) -> dict:
    """Заглушка модели: класс выбирается по ключевому слову в тексте."""
    lowered = text.lower()
    if "отличн" in lowered:
        label = "positive"
    elif "ужасн" in lowered:
        label = "negative"
    else:
        label = "neutral"
    probs = CANNED[label]
    return {"label": label, "score": probs[label], "probabilities": dict(probs)}


@pytest.fixture
def model_calls():
    """Список текстов, которые API передал в модель."""
    return []


@pytest.fixture
def client(monkeypatch, model_calls):
    def recording_fake(text: str) -> dict:
        model_calls.append(text)
        return fake_prediction(text)

    monkeypatch.setattr(service, "warm_up", lambda: None)
    monkeypatch.setattr(service, "analyze", recording_fake)
    # with-блок запускает lifespan приложения
    with TestClient(app) as test_client:
        yield test_client
