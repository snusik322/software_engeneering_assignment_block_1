from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

MAX_TEXT_LENGTH = 5000
MAX_BATCH_SIZE = 20

# Текст без пробелов по краям, длиной от 1 до MAX_TEXT_LENGTH символов.
# Строка из одних пробелов после обрезки становится пустой и отклоняется.
ReviewText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH),
]


class SentimentRequest(BaseModel):
    text: ReviewText = Field(
        description="Текст отзыва на русском языке",
        examples=["Фильм посмотрел на одном дыхании, актёры сыграли великолепно"],
    )


class BatchRequest(BaseModel):
    texts: list[ReviewText] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"От 1 до {MAX_BATCH_SIZE} отзывов",
    )


class SentimentResponse(BaseModel):
    label: Literal["negative", "neutral", "positive"]
    score: float = Field(ge=0, le=1, description="Вероятность выбранного класса")
    probabilities: dict[str, float] = Field(description="Вероятности всех классов")


class BatchResponse(BaseModel):
    results: list[SentimentResponse] = Field(description="Результаты в том же порядке, что и тексты в запросе")


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
