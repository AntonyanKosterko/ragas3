# 🤖 RAG Система для Исследований

Комплексный и модульный проект RAG (Retrieval-Augmented Generation) на Python для проведения исследований. Проект полностью готов к запуску, включает систему оценки моделей, трекинг экспериментов и демонстрационный веб-интерфейс.

## 🚀 Ключевые особенности

- **Модульность**: Весь пайплайн построен из заменяемых блоков (модели, сплиттеры, ретриверы)
- **Конфигурируемость**: Параметры экспериментов задаются через YAML конфигурации
- **Воспроизводимость**: Полное логирование параметров и метрик для анализа
- **Поддержка железа**: Автоматическое определение GPU/CPU и оптимизация под доступные ресурсы
- **Система оценки**: Комплексные метрики качества (косинусное сходство, ROUGE, BLEU)
- **MLflow интеграция**: Трекинг экспериментов и сравнение результатов
- **Веб-интерфейс**: Интерактивные интерфейсы на Gradio и Streamlit

## 📁 Структура проекта

```
RAGAS/
├── config/                  # Конфигурационные файлы
│   ├── base_config.yaml    # Базовая конфигурация
│   ├── gpu_config.yaml     # Конфигурация для GPU
│   └── cpu_config.yaml     # Конфигурация для CPU
├── data/                   # Данные и датасеты
│   └── documents/          # Исходные документы
├── src/                    # Исходный код
│   ├── __init__.py
│   ├── models.py           # Загрузка и управление моделями
│   ├── data_processing.py  # Обработка данных
│   ├── pipeline.py         # Основной RAG пайплайн
│   └── evaluation.py       # Система оценки
├── notebooks/              # Jupyter ноутбуки для экспериментов
├── logs/                   # Логи экспериментов
├── results/                # Результаты оценки
├── mlruns/                 # MLflow эксперименты
├── main.py                 # Основной скрипт с MLflow
├── app.py                  # Веб-интерфейс
├── requirements.txt        # Зависимости
└── README.md              # Документация
```

## 🛠️ Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd RAGAS
```

### 2. Создание виртуального окружения

```bash
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# или
rag_env\Scripts\activate     # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Создание необходимых директорий

```bash
mkdir -p data/documents logs results notebooks
```

## ⚙️ Конфигурация

Проект поддерживает три предустановленные конфигурации:

### Базовая конфигурация (`config/base_config.yaml`)
- Универсальная настройка для большинства случаев
- Автоматическое определение устройства (GPU/CPU)
- Сбалансированные параметры

### GPU конфигурация (`config/gpu_config.yaml`)
- Оптимизирована для GPU с VRAM до 24GB
- Использует большие модели для лучшего качества
- Ускоренная обработка

### CPU конфигурация (`config/cpu_config.yaml`)
- Оптимизирована для CPU
- Использует квантизованные модели
- Минимальные требования к ресурсам

## 🚀 Быстрый старт

### 1. Подготовка данных

Поместите ваши документы в папку `data/documents/`. Поддерживаются форматы:
- `.txt` - текстовые файлы
- `.pdf` - PDF документы
- `.md` - Markdown файлы

### 2. Запуск эксперимента

```bash
# Базовый эксперимент
python main.py --config config/base_config.yaml

# GPU эксперимент
python main.py --config config/gpu_config.yaml

# CPU эксперимент
python main.py --config config/cpu_config.yaml

# Принудительное пересоздание векторной базы
python main.py --config config/base_config.yaml --force-rebuild

# Без оценки (только создание пайплайна)
python main.py --config config/base_config.yaml --no-evaluation
```

### 3. Запуск веб-интерфейса

#### Gradio интерфейс
```bash
python app.py --config config/base_config.yaml --interface gradio
```

#### Streamlit интерфейс
```bash
# Через Python
python app.py --config config/base_config.yaml --interface streamlit

# Или напрямую через Streamlit
streamlit run app.py -- --config config/base_config.yaml --interface streamlit
```

## 📊 Система оценки

### Поддерживаемые метрики

1. **Косинусное сходство** - семантическая схожесть ответов
2. **ROUGE** - метрики для оценки качества текста
3. **BLEU** - метрика для оценки качества перевода/генерации
4. **Exact Match** - точное совпадение ответов
5. **Length Ratio** - соотношение длин ответов

### Запуск оценки

```bash
# Оценка с сохранением результатов
python main.py --config config/base_config.yaml

# Результаты сохраняются в results/
```

### Сравнение экспериментов

```bash
# Сравнение нескольких запусков
python main.py --compare run_id_1 run_id_2 run_id_3
```

## 🔬 MLflow интеграция

### Просмотр экспериментов

```bash
# Запуск MLflow UI
mlflow ui

# Откройте http://localhost:5000 в браузере
```

### Логируемые параметры

- **Модели**: Названия и настройки embedding и generator моделей
- **Данные**: Параметры чанкинга и сплиттера
- **Векторная база**: Тип и настройки хранилища
- **Ретривер**: Параметры поиска документов

### Логируемые метрики

- **Качество**: Все метрики оценки (cosine_similarity, rouge, bleu)
- **Производительность**: Время ответа, использование памяти
- **Статистика**: Количество документов, запросов

## 🌐 Веб-интерфейс

### Gradio интерфейс

- Интерактивный чат
- Отображение источников информации
- Информация о системе в реальном времени
- Примеры вопросов

### Streamlit интерфейс

- Чистый и современный дизайн
- Боковая панель с метриками
- Развернутая информация об источниках
- Адаптивная верстка

## 🔧 Настройка под ваше железо

### Для GPU (NVIDIA)

1. Убедитесь, что у вас установлен CUDA
2. Используйте `config/gpu_config.yaml`
3. Модели автоматически загрузятся на GPU

### Для CPU

1. Используйте `config/cpu_config.yaml`
2. Модели будут квантизованы для экономии памяти
3. Рекомендуется минимум 8GB RAM

### Кастомная конфигурация

Создайте свой конфигурационный файл на основе `base_config.yaml`:

```yaml
models:
  embedding:
    name: "your-embedding-model"
    device: "cuda"  # или "cpu"
  generator:
    name: "your-generator-model"
    device: "cuda"
    # Дополнительные параметры...
```

## 📈 Мониторинг и отладка

### Логи

Все логи сохраняются в папке `logs/`:
- `rag_experiment.log` - основные логи
- `chainlit_client.log` - логи веб-интерфейса

### Статистика пайплайна

```python
from src.pipeline import create_rag_pipeline
import yaml

# Загружаем конфигурацию
with open('config/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Создаем пайплайн
pipeline = create_rag_pipeline(config)
pipeline.initialize()

# Получаем статистику
stats = pipeline.get_stats()
print(f"Всего запросов: {stats['total_queries']}")
print(f"Среднее время ответа: {stats['avg_response_time']:.2f} сек")
```

## 🧪 Примеры использования

### Базовое использование

```python
from src.pipeline import create_rag_pipeline
import yaml

# Загружаем конфигурацию
with open('config/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Создаем и инициализируем пайплайн
pipeline = create_rag_pipeline(config)
pipeline.initialize()

# Задаем вопрос
question = "Что такое машинное обучение?"
result = pipeline.query(question)

print(f"Вопрос: {question}")
print(f"Ответ: {result['answer']}")
print(f"Время ответа: {result['response_time']:.2f} сек")
```

### Оценка качества

```python
from src.evaluation import create_evaluator
from src.data_processing import create_data_processor

# Создаем оценщик
evaluator = create_evaluator(config)
data_processor = create_data_processor(config)

# Загружаем датасет
qa_dataset = data_processor.load_qa_dataset('data/russian_qa_dataset.json')

# Запускаем оценку
results = evaluator.evaluate_pipeline(pipeline, qa_dataset)

# Выводим метрики
for metric, value in results['metrics'].items():
    if metric.endswith('_mean'):
        print(f"{metric}: {value:.4f}")
```

### Поиск похожих документов

```python
# Поиск похожих документов
similar_docs = pipeline.similarity_search("машинное обучение", k=5)

for i, doc in enumerate(similar_docs, 1):
    print(f"Документ {i}:")
    print(f"Содержимое: {doc['content'][:200]}...")
    print(f"Релевантность: {doc['score']:.4f}")
    print()
```

## 🐛 Устранение неполадок

### Проблемы с памятью

1. **GPU**: Используйте `config/cpu_config.yaml` или уменьшите размер чанков
2. **CPU**: Уменьшите `chunk_size` в конфигурации

### Проблемы с моделями

1. Проверьте подключение к интернету для загрузки моделей
2. Убедитесь, что у вас достаточно места на диске
3. Проверьте совместимость версий PyTorch и CUDA

### Проблемы с векторной базой

1. Удалите папку `data/vector_db*` и пересоздайте базу
2. Проверьте права доступа к папке
3. Убедитесь, что документы загружены в `data/documents/`

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

## 📞 Поддержка

Если у вас возникли вопросы или проблемы:

1. Проверьте раздел "Устранение неполадок"
2. Создайте Issue в репозитории
3. Обратитесь к документации MLflow и LangChain

## 🔄 Обновления

### v1.0.0
- Первоначальный релиз
- Поддержка GPU и CPU
- Интеграция с MLflow
- Веб-интерфейсы на Gradio и Streamlit
- Комплексная система оценки

---

**Удачных экспериментов с RAG! 🚀**

