# Деплой ИИ-моделей

Для работы я выбрал 4 простеньких задач, на которых удобно разобраться, как пользоваться готовыми ML-моделями:
определение тональности текста, классификация изображений, классификация звуков, детекция объектов на видео. В каждой задаче я проверил как модель работает на моих примерах и посчитал метрики качества.


| № | Задача | Фреймворк | Модель | Результат |
|---|--------|-----------|--------|-----------|
| 1 | Тональность отзывов о фильмах | Hugging Face | blanchefort/rubert-base-cased-sentiment | Accuracy 0.722 |
| 2 | Классификация изображений | PyTorch (torchvision) | ResNet-50 | Top-1 0.900 / Top-5 1.000 |
| 3 | Классификация звуков | TensorFlow Hub | YAMNet | Top-1 0.455  / Top-5 1.000  |
| 4 | Детекция объектов на видео | Hugging Face | DETR ResNet-50 | Верных кадров 0.820 |

Использовано три фреймворка: Hugging Face, PyTorch, TensorFlow Hub.

## Установка
### 1. Создание виртуального окружения
macOS / Linux: python3 -m venv .venv

Windows: python -m venv .venv

### 2. Активация виртуального окружения
macOS / Linux: source .venv/bin/activate

Windows: .venv\Scripts\activate

### 3. Установка зависимостей
pip install -r requirements.txt

# Часть 1. ML-модели

## Определение тональности текста

Запуск: python text/sentiment.py "Отличный фильм"

Тестовые отзывы находятся в: text/data/test_samples.json

Запуск собственного тестирования: python text/evaluate.py

## Классификация изображений

Запуск: python image/classify.py путь_к_фото.jpg

Тестовые изображения находятся в: image/data/

Запуск собственного тестирования: python image/evaluate.py

## Классификация звуков

Запуск: python audio/yamnet_classify.py путь_к_записи.wav

Тестовые аудиофайлы находятся в: audio/data/

Запуск собственного тестирования: python audio/evaluate.py

## Детекция объектов на видео

Запуск: python 4_video/detect_video.py путь_к_видео.mp4 --out 4_video/output/result.mp4

Тестовые видео находятся в: 4_video/data/

Запуск собственного тестирования: python 4_video/evaluate.py

# Используемые технологии

- Python
- PyTorch
- TensorFlow / TensorFlow Hub
- Hugging Face Transformers
- torchvision
- OpenCV


# Часть 2. API для ML-модели

Для модели определения тональности текста разработан REST API на FastAPI. API позволяет определить тональность одного или нескольких отзывов и проверить состояние загруженной модели.

### Запуск API

```bash
uvicorn api.main:app --reload
```

После запуска доступны:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Проверка состояния API: `GET /health`
- Определение тональности текста: `POST /sentiment`
- Пакетное определение тональности: `POST /sentiment/batch`

Пример запроса:

```json
{
  "text": "Отличный фильм, очень понравился!"
}
```

API возвращает определённую тональность (`positive`, `neutral` или `negative`), вероятность выбранного класса и вероятности всех классов.

### Тестирование API

Для тестирования используется PyTest. Тесты проверяют работу endpoint'ов, валидацию входных данных, обработку ошибок и корректность ответов модели.

Запуск всех тестов:

```bash
pytest -v
```

Тесты с подменённой моделью можно запустить отдельно:

```bash
pytest -m "not model" -v
```

Тесты с настоящей ML-моделью:

```bash
pytest -m model -v
```

### GitHub Actions

Для автоматического запуска тестов настроен GitHub Actions. Тесты автоматически запускаются при `push` и создании `pull request`.

Workflow находится в:

```text
.github/workflows/test.yml
```