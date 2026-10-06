import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api import service
from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
)

log = logging.getLogger("sentiment-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель грузится один раз при старте, а не при первом запросе:
    # в части 1 загрузка занимала около 3 с, а одно предсказание около 24 мс.
    service.warm_up()
    yield


app = FastAPI(
    title="Sentiment API",
    description="Определение тональности русскоязычных отзывов (negative / neutral / positive).",
    version="1.0.0",
    lifespan=lifespan,
)


def run_model(text: str) -> dict:
    """Вызывает модель и переводит её ошибки в корректные HTTP-ответы."""
    try:
        return service.analyze(text)
    except ValueError as exc:
        # некорректные данные от клиента
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        # сбой на стороне сервиса: подробности только в логе, клиенту общий ответ
        log.exception("Ошибка при работе модели")
        raise HTTPException(status_code=500, detail="Не удалось обработать текст")


@app.get("/health", response_model=HealthResponse, summary="Проверка работоспособности")
def health():
    return HealthResponse(status="ok", model=service.MODEL_NAME, model_loaded=service.is_ready())


@app.post("/sentiment", response_model=SentimentResponse, summary="Тональность одного отзыва")
def sentiment(request: SentimentRequest):
    return run_model(request.text)


@app.post("/sentiment/batch", response_model=BatchResponse, summary="Тональность нескольких отзывов")
def sentiment_batch(request: BatchRequest):
    return BatchResponse(results=[run_model(text) for text in request.texts])
