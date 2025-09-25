# 🚀 RAG Система с датасетом SberQuAD

## 📋 Описание

RAG (Retrieval-Augmented Generation) система для работы с русскоязычными вопросами и ответами на основе датасета SberQuAD.

### 🎯 Особенности:
- **Датасет**: SberQuAD (45,328 примеров, 5,933 уникальных документа)
- **Модели**: Оптимизированы для CPU
- **Интерфейс**: Gradio веб-интерфейс
- **Тестирование**: Автоматическое тестирование с метриками
- **MLflow**: Отслеживание экспериментов

## 🚀 Быстрый запуск

### 1. Загрузка датасета
```bash
python load_sberquad.py
```

### 2. Тестирование системы
```bash
python test_rag.py --config config/base_config.yaml --max-samples 100
```

### 3. Веб-интерфейс
```bash
python app.py --config config/base_config.yaml --interface gradio
```
Откройте браузер: http://localhost:7860

### 4. MLflow UI
```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

## 📁 Структура проекта

```
RAGAS/
├── load_sberquad.py          # Загрузчик датасета SberQuAD
├── test_rag.py              # Тестер RAG системы
├── app.py                   # Веб-интерфейс
├── main.py                  # Эксперименты с MLflow
├── config/
│   └── base_config.yaml     # Конфигурация
├── src/                     # Исходный код
├── datasets/sberquad/       # Датасет SberQuAD
│   ├── documents_for_rag.json    # 500 документов
│   ├── qa_pairs.json            # 200 вопросов-ответов
│   └── vector_db/               # Векторная БД
└── results/                 # Результаты тестирования
```

## ⚙️ Конфигурация

Основные настройки в `config/base_config.yaml`:

- **Эмбеддинг**: `all-MiniLM-L6-v2` (быстрая модель для CPU)
- **Генератор**: Кастомная RAG модель
- **Векторная БД**: FAISS (оптимизирована для CPU)
- **Ретривер**: 5 документов для поиска

## 📊 Метрики тестирования

Система автоматически вычисляет:
- **ROUGE** (1, 2, L) - качество генерации
- **BLEU** - соответствие эталону
- **Exact Match** - точное совпадение
- **Response Time** - время ответа
- **Source Coverage** - покрытие источников

## 🔧 Требования

- Python 3.8+
- 8GB RAM (рекомендуется)
- CPU (GPU не требуется)

## 📈 Результаты

После тестирования результаты сохраняются в:
- `results/rag_test_results.json` - детальные результаты
- MLflow UI - визуализация экспериментов

## 🎯 Примеры использования

### Тестирование с разным количеством примеров:
```bash
# Быстрый тест (10 примеров)
python test_rag.py --config config/base_config.yaml --max-samples 10

# Полный тест (100 примеров)
python test_rag.py --config config/base_config.yaml --max-samples 100

# Пересоздание векторной БД
python test_rag.py --config config/base_config.yaml --rebuild-vector-db
```

### Веб-интерфейс:
- Откройте http://localhost:7860
- Введите вопрос на русском языке
- Получите ответ на основе документов SberQuAD

## 📝 Логи

Логи сохраняются в `logs/` директории для отладки и мониторинга.

## 🤝 Поддержка

При возникновении проблем проверьте:
1. Установлены ли все зависимости: `pip install -r requirements.txt`
2. Загружен ли датасет: `python load_sberquad.py`
3. Создана ли векторная БД: `python test_rag.py --rebuild-vector-db`