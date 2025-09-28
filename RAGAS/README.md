# 🚀 RAG Система с датасетом SberQuAD

## 📋 Описание

RAG (Retrieval-Augmented Generation) система для работы с русскоязычными вопросами и ответами на основе датасета SberQuAD.

### 🎯 Особенности:
- **Датасет**: SberQuAD (2000 документов, 500 вопросов-ответов)
- **Модели**: Многоязычная модель эмбеддингов + кастомная RAG генератор
- **Интерфейс**: Gradio веб-интерфейс
- **Тестирование**: Автоматическое тестирование с метриками ретривера и генерации
- **MLflow**: Отслеживание экспериментов

## 🚀 Быстрый запуск

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Подготовка датасета
```bash
python load_sberquad.py
```

### 3. Тестирование системы
```bash
# Быстрый тест (10 примеров)
python test_rag.py --config config/cpu_optimized_config.yaml --max-samples 10

# Полный тест (500 примеров)
python test_rag.py --config config/cpu_optimized_config.yaml --max-samples 500
```

### 4. Веб-интерфейс
```bash
python app.py --config config/cpu_optimized_config.yaml --interface gradio
```
Откройте браузер: http://localhost:7860

### 5. MLflow UI
```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```
Откройте браузер: http://localhost:5000

## 📁 Структура проекта

```
RAGAS/
├── load_sberquad.py              # Загрузчик датасета SberQuAD
├── test_rag.py                   # Тестер RAG системы
├── app.py                        # Веб-интерфейс
├── main.py                       # Эксперименты с MLflow
├── config/
│   ├── cpu_optimized_config.yaml # Конфигурация для CPU (рекомендуется)
│   ├── gpu_optimal_config.yaml   # Конфигурация для GPU
│   └── base_config.yaml          # Базовая конфигурация
├── src/                          # Исходный код
│   ├── pipeline.py               # RAG пайплайн
│   ├── evaluation.py             # Система оценки
│   ├── models.py                 # Управление моделями
│   ├── rag_generator.py          # Кастомная RAG генератор
│   └── dataset_loader.py         # Загрузка данных
├── datasets/sberquad/            # Датасет SberQuAD
│   ├── documents_for_rag.json    # 2000 документов
│   ├── qa_pairs.json             # 500 вопросов-ответов
│   └── vector_db/                # Векторная БД
├── results/                      # Результаты тестирования
├── mlruns/                       # MLflow эксперименты
└── RETRIEVER_METRICS.md          # Документация метрик ретривера
```

## ⚙️ Конфигурации

### CPU конфигурация (рекомендуется)
- **Файл**: `config/cpu_optimized_config.yaml`
- **Эмбеддинг**: `paraphrase-multilingual-MiniLM-L12-v2` (многоязычная)
- **Генератор**: Кастомная RAG модель
- **Векторная БД**: FAISS (оптимизирована для CPU)
- **Ретривер**: 3 документа для поиска

### GPU конфигурация
- **Файл**: `config/gpu_optimal_config.yaml`
- **Эмбеддинг**: `paraphrase-multilingual-mpnet-base-v2` (мощная многоязычная)
- **Генератор**: `IlyaGusev/saiga_llama3_8b` (Saiga-7B)
- **Векторная БД**: FAISS
- **Ретривер**: 5 документов для поиска

## 📊 Метрики тестирования

### Метрики генерации:
- **ROUGE** (1, 2, L) - качество генерации
- **BLEU** - соответствие эталону
- **Exact Match** - точное совпадение
- **Cosine Similarity** - семантическое сходство
- **Response Time** - время ответа

### Метрики ретривера:
- **Precision@K** - точность извлечения документов
- **Recall@K** - полнота извлечения
- **F1@K** - баланс точности и полноты
- **Hit Rate@K** - процент успешных поисков
- **MRR** - средний обратный ранг
- **NDCG@K** - нормализованный DCG
- **Coverage** - покрытие релевантных документов
- **Diversity** - разнообразие извлеченных документов

## 🎯 Примеры использования

### Тестирование с разным количеством примеров:
```bash
# Быстрый тест (10 примеров)
python test_rag.py --config config/cpu_optimized_config.yaml --max-samples 10

# Полный тест (500 примеров)
python test_rag.py --config config/cpu_optimized_config.yaml --max-samples 500

# Пересоздание векторной БД
python test_rag.py --config config/cpu_optimized_config.yaml --rebuild-vector-db
```

### Веб-интерфейс:
- Откройте http://localhost:7860
- Введите вопрос на русском языке
- Получите ответ на основе документов SberQuAD

### Эксперименты с MLflow:
```bash
python main.py --config config/cpu_optimized_config.yaml
```

## 📈 Результаты

После тестирования результаты сохраняются в:
- `results/rag_test_results.json` - детальные результаты
- MLflow UI - визуализация экспериментов

### Типичные результаты (CPU конфигурация):
- **Hit Rate**: 74.8% (успешных поисков)
- **Precision**: 24.9% (точность извлечения)
- **Recall**: 74.8% (полнота извлечения)
- **MRR**: 67.2% (качество ранжирования)
- **Время ответа**: ~44мс

## 🔧 Требования

- Python 3.8+
- 8GB RAM (рекомендуется)
- CPU (GPU не требуется для CPU конфигурации)

## 🛠️ Устранение проблем

### Если система работает медленно:
1. Используйте `cpu_optimized_config.yaml`
2. Уменьшите `max_samples` в тестировании
3. Закройте другие приложения

### Если не хватает памяти:
1. Уменьшите `batch_size` в конфигурации
2. Используйте меньше документов в датасете
3. Закройте другие приложения

### Если веб-интерфейс не запускается:
1. Проверьте, что порт 7860 свободен
2. Попробуйте другой порт в конфигурации
3. Используйте `--interface streamlit` вместо gradio

### Если метрики ретривера нулевые:
1. Убедитесь, что используется многоязычная модель эмбеддингов
2. Пересоздайте векторную БД: `--rebuild-vector-db`
3. Проверьте, что датасет загружен корректно

## 📝 Логи

Логи сохраняются в `logs/` директории для отладки и мониторинга.

## 🤝 Поддержка

При возникновении проблем проверьте:
1. Установлены ли все зависимости: `pip install -r requirements.txt`
2. Загружен ли датасет: `python load_sberquad.py`
3. Создана ли векторная БД: `python test_rag.py --rebuild-vector-db`
4. Правильная ли конфигурация: используйте `cpu_optimized_config.yaml`

## 📚 Дополнительная документация

- `RETRIEVER_METRICS.md` - подробное описание метрик ретривера
- MLflow UI - визуализация экспериментов и метрик
- `results/` - детальные результаты тестирования