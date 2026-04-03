# HW13 — токенизация, BERT-инференс и fine-tuning для классификации текста

## 1. Кратко: что сделано

- Загружен датасет `emotion` (6 классов эмоций).
- Показана токенизация: примеры токенов, input_ids, attention_mask, special tokens.
- Проведён инференс готовой моделью `bert-base-uncased` без обучения (результаты случайные).
- Выполнен fine-tuning `bert-base-uncased` под задачу классификации эмоций.
- Оценка на тестовой выборке: accuracy 92.35%, F1-macro 0.8867.
- Артефакты сохранены в `./artifacts/`.

## 2. Среда и воспроизводимость

- Python 3.11, PyTorch 2.10.0 (CPU)
- Seed: 42
- Запуск: открыть `HW13.ipynb` и выполнить Run All

## 3. Данные

- Датасет: `emotion` (HuggingFace)
- Train: 16000, Validation: 2000, Test: 2000
- Классы: sadness, joy, love, anger, fear, surprise

## 4. Модель и обучение

- База: `bert-base-uncased`
- Токенизация: `AutoTokenizer`, max_length=128
- Обучение: ручной цикл (DataLoader + AdamW), 3 эпохи, batch_size=32
- Выбор лучшей модели по validation accuracy

## 5. Результаты

- Test accuracy: 92.35%
- Test F1-macro: 0.8867
- Лучше всего распознаются: sadness (0.96), joy (0.94)
- Хуже всего: surprise (0.78), love (0.82)

**Артефакты:**
- Матрица ошибок: `./artifacts/confusion_matrix.png`
- Примеры предсказаний: `./artifacts/sample_predictions.csv`

## 6. Итоговый вывод

Готовая модель без обучения бесполезна для новой задачи — предсказания случайны. Fine-tuning дал отличный результат: 92% accuracy. Токенизация — обязательный шаг, без неё модель не поймёт текст. Для ускорения стоит использовать GPU или более лёгкую модель (DistilBERT).