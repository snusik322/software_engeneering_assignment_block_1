import pytest
from fastapi.testclient import TestClient
from api import service
from api.main import app
from text import sentiment

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def real_client():
    # модель загружается один раз на весь модуль
    with TestClient(app) as test_client:
        yield test_client


def test_model_name_matches_part_one():
    assert service.MODEL_NAME == sentiment.MODEL_NAME


def test_health_shows_loaded_model(real_client):
    body = real_client.get("/health").json()
    assert body["model_loaded"] is True


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Фильм посмотрел на одном дыхании, актёры сыграли великолепно", "positive"),
        ("Фильм длится два часа десять минут, снят в 2019 году", "neutral"),
        ("Ужасный сюжет, дыры на каждом шагу, зря потратил вечер", "negative"),
    ],
    ids=["positive", "neutral", "negative"],
)
def test_real_model_labels(real_client, text, expected):
    body = real_client.post("/sentiment", json={"text": text}).json()
    assert body["label"] == expected


def test_real_probabilities_are_consistent(real_client):
    body = real_client.post("/sentiment", json={"text": "Сеанс начинается в девять вечера"}).json()
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert body["score"] == max(body["probabilities"].values())


def test_batch_matches_single_requests(real_client):
    texts = [
        "Фильм посмотрел на одном дыхании, актёры сыграли великолепно",
        "Ужасный сюжет, дыры на каждом шагу, зря потратил вечер",
    ]
    batch = real_client.post("/sentiment/batch", json={"texts": texts}).json()["results"]
    single = [real_client.post("/sentiment", json={"text": t}).json() for t in texts]
    assert [r["label"] for r in batch] == [r["label"] for r in single]
