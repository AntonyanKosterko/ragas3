# 🚀 Быстрый старт RAG системы

Этот файл содержит краткие инструкции для быстрого запуска RAG системы.

## ⚡ Мгновенный запуск

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Запуск базового эксперимента
```bash
python main.py --config config/base_config.yaml
```

### 3. Запуск веб-интерфейса
```bash
# Gradio интерфейс
python app.py --config config/base_config.yaml --interface gradio

# Streamlit интерфейс  
python app.py --config config/base_config.yaml --interface streamlit
```

## 🔧 Конфигурации по умолчанию

| Конфигурация | Описание | Использование |
|-------------|----------|---------------|
| `base_config.yaml` | Универсальная | `python main.py` |
| `gpu_config.yaml` | Для GPU | `python main.py --config config/gpu_config.yaml` |
| `cpu_config.yaml` | Для CPU | `python main.py --config config/cpu_config.yaml` |

## 📊 Просмотр результатов

### MLflow UI
```bash
mlflow ui
# Откройте http://localhost:5000
```

### Результаты оценки
- JSON: `results/evaluation_*.json`
- CSV: `results/evaluation_*_predictions.csv`

## 🧪 Примеры использования

```bash
# Запуск всех примеров
python examples.py

# Jupyter ноутбук
jupyter notebook notebooks/example_experiment.ipynb
```

## 🐛 Быстрое решение проблем

### Проблема: Не хватает памяти
```bash
# Используйте CPU конфигурацию
python main.py --config config/cpu_config.yaml
```

### Проблема: Модели не загружаются
```bash
# Проверьте интернет-соединение
# Убедитесь, что у вас достаточно места на диске
```

### Проблема: Векторная база не создается
```bash
# Принудительно пересоздайте базу
python main.py --config config/base_config.yaml --force-rebuild
```

## 📁 Структура файлов

```
RAGAS/
├── config/          # Конфигурации
├── data/            # Данные
├── src/             # Исходный код
├── notebooks/       # Jupyter ноутбуки
├── results/         # Результаты
├── logs/            # Логи
├── main.py          # Основной скрипт
├── app.py           # Веб-интерфейс
└── examples.py      # Примеры
```

## 🎯 Основные команды

```bash
# Эксперимент
python main.py --config config/base_config.yaml

# Веб-интерфейс
python app.py --config config/base_config.yaml

# Примеры
python examples.py

# MLflow UI
mlflow ui

# Помощь
python main.py --help
python app.py --help
```

## 📞 Поддержка

- 📖 Полная документация: `README.md`
- 🧪 Примеры: `examples.py`
- 📓 Ноутбуки: `notebooks/`
- 🐛 Проблемы: проверьте `logs/`

---

**Готово! Начинайте эксперименты! 🎉**

