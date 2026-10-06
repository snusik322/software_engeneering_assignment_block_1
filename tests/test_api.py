import pytest

from api import service
from api.schemas import MAX_BATCH_SIZE, MAX_TEXT_LENGTH

LABELS = {"negative", "neutral", "positive"}


# служебные адреса

def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == service.MODEL_NAME
    assert isinstance(body["model_loaded"], bool)


def test_openapi_lists_all_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/sentiment", "/sentiment/batch"} <= set(paths)


def test_unknown_path_returns_404(client):
    assert client.get("/no-such-page").status_code == 404


def test_get_on_post_endpoint_returns_405(client):
    assert client.get("/sentiment").status_code == 405


# один отзыв: формат и содержимое ответа

def test_response_has_documented_fields(client):
    response = client.post("/sentiment", json={"text": "Отличный фильм"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "score", "probabilities"}
    assert body["label"] in LABELS
    assert 0.0 <= body["score"] <= 1.0
    assert set(body["probabilities"]) == LABELS


def test_probabilities_sum_to_one(client):
    body = client.post("/sentiment", json={"text": "Сеанс в девять вечера"}).json()
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=0.01)


def test_score_equals_probability_of_label(client):
    body = client.post("/sentiment", json={"text": "Отличный фильм"}).json()
    assert body["score"] == body["probabilities"][body["label"]]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Отличный фильм", "positive"),
        ("Ужасный сюжет", "negative"),
        ("Сеанс в девять вечера", "neutral"),
    ],
)
def test_label_from_model_is_returned_unchanged(client, text, expected):
    assert client.post("/sentiment", json={"text": text}).json()["label"] == expected


def test_surrounding_spaces_are_stripped_before_model(client, model_calls):
    client.post("/sentiment", json={"text": "   Отличный фильм \n"})
    assert model_calls == ["Отличный фильм"]


def test_text_of_maximum_length_is_accepted(client):
    response = client.post("/sentiment", json={"text": "а" * MAX_TEXT_LENGTH})
    assert response.status_code == 200


# некорректные запросы 

@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},                         # пустая строка
        {"text": "   \n\t "},                 # только пробельные символы
        {"text": "а" * (MAX_TEXT_LENGTH + 1)},  # слишком длинный текст
        {},                                   # нет обязательного поля
        {"text": None},                       # null вместо строки
        {"text": 12345},                      # число вместо строки
        {"text": ["Отличный фильм"]},         # список вместо строки
    ],
    ids=["empty", "whitespace", "too-long", "no-field", "null", "number", "list"],
)
def test_invalid_payload_is_rejected_with_422(client, payload, model_calls):
    response = client.post("/sentiment", json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert model_calls == []   # до модели некорректный запрос не доходит


def test_malformed_json_is_rejected_with_422(client):
    response = client.post(
        "/sentiment", content=b"{not a json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# ошибки модели

def test_value_error_from_model_becomes_422(client, monkeypatch):
    def refuse(text):
        raise ValueError("Текст не может быть пустым")

    monkeypatch.setattr(service, "analyze", refuse)
    response = client.post("/sentiment", json={"text": "Отличный фильм"})
    assert response.status_code == 422
    assert "пустым" in response.json()["detail"]


def test_unexpected_model_failure_becomes_500_without_details(client, monkeypatch):
    def crash(text):
        raise RuntimeError("секретная внутренняя причина")

    monkeypatch.setattr(service, "analyze", crash)
    response = client.post("/sentiment", json={"text": "Отличный фильм"})
    assert response.status_code == 500
    assert "секретная" not in response.text


# пакетная обработка

def test_batch_keeps_order_and_count(client):
    texts = ["Отличный фильм", "Ужасный сюжет", "Сеанс в девять вечера"]
    response = client.post("/sentiment/batch", json={"texts": texts})
    assert response.status_code == 200
    labels = [item["label"] for item in response.json()["results"]]
    assert labels == ["positive", "negative", "neutral"]


def test_batch_accepts_maximum_size(client):
    response = client.post("/sentiment/batch", json={"texts": ["Отличный фильм"] * MAX_BATCH_SIZE})
    assert response.status_code == 200
    assert len(response.json()["results"]) == MAX_BATCH_SIZE


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},                                   # пустой список
        {"texts": ["текст"] * (MAX_BATCH_SIZE + 1)},     # слишком много текстов
        {"texts": ["Отличный фильм", "   "]},            # один из текстов пустой
        {"texts": "Отличный фильм"},                     # строка вместо списка
        {},                                              # нет поля texts
    ],
    ids=["empty-list", "too-many", "blank-item", "not-a-list", "no-field"],
)
def test_invalid_batch_is_rejected_with_422(client, payload):
    assert client.post("/sentiment/batch", json=payload).status_code == 422
