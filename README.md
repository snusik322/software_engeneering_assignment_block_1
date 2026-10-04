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

Запуск: python 1_text/sentiment.py "Отличный фильм"

Тестовые отзывы находятся в: 1_text/data/test_samples.json

Запуск собственного тестирования: python 1_text/evaluate.py

## Классификация изображений

Запуск: python 2_image/classify.py путь_к_фото.jpg

Тестовые изображения находятся в: 2_image/data/

Запуск собственного тестирования: python 2_image/evaluate.py

## Классификация звуков

Запуск: python 3_audio/yamnet_classify.py путь_к_записи.wav

Тестовые аудиофайлы находятся в: 3_audio/data/

Запуск собственного тестирования: python 3_audio/evaluate.py

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