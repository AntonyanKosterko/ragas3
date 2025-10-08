# 🔧 Руководство по оптимизации гиперпараметров ретривера

## 📋 Обзор системы

Система оптимизации гиперпараметров использует **байесовский поиск** (TPE - Tree-structured Parzen Estimator) для эффективного поиска оптимальных параметров ретривера. Каждое испытание включает полный тест RAG системы с вычислением всех метрик качества.

## ✅ Что работает

### 1. **Прогресс-бар в терминале**
```
Оптимизация: 100%|█████████████████████████████████████████| 2/2 [00:13<00:00, 6.84s/испытание, best_score=0.0000, current_score=-0.0100]
```

### 2. **MLflow эксперименты**
- Эксперименты создаются автоматически
- Логируются параметры и метрики
- Результаты сохраняются в `mlruns/`

### 3. **Байесовский поиск**
- Использует TPE (Tree-structured Parzen Estimator)
- Эффективно исследует пространство параметров
- Автоматически обрезает плохие испытания

## 🚀 Быстрый старт

### 1. Запуск оптимизации
```bash
# Упрощенная версия с прогресс-баром
python simple_hyperparameter_optimization.py --n-trials 10 --timeout 1800

# Полная версия (если исправлена)
python hyperparameter_optimization.py --n-trials 20 --timeout 3600
```

### 2. Параметры командной строки
```bash
python simple_hyperparameter_optimization.py \
    --config config/hybrid_cpu_config.yaml \
    --n-trials 10 \
    --timeout 1800 \
    --experiment-name "My_Retriever_Optimization"
```

## 🎯 Оптимизируемые гиперпараметры

### 1. **Веса гибридного поиска**
- `semantic_weight`: 0.3 - 0.9 (шаг 0.1)
- `bm25_weight`: автоматически = 1.0 - semantic_weight

### 2. **Количество документов**
- `semantic_k`: 8 - 20 (шаг 2)
- `bm25_k`: 8 - 20 (шаг 2)  
- `final_k`: 3 - 10 (шаг 1)

### 3. **Параметры BM25**
- `bm25_k1`: 0.8 - 2.0 (шаг 0.1)
- `bm25_b`: 0.5 - 1.0 (шаг 0.05)

### 4. **Размер чанков**
- `chunk_size`: [300, 400, 500, 600, 700, 800]
- `chunk_overlap`: 30 - 100 (шаг 10)

### 5. **Тип поиска**
- `search_type`: ['hybrid', 'similarity', 'bm25']

## 📊 Целевая функция

Система оптимизирует **комбинированную оценку качества**:

```python
quality_score = 0.4 * MRR + 0.2 * Precision + 0.2 * Recall + 0.1 * NDCG + 0.1 * Hit_Rate
time_penalty = min(0.1, retrieval_time * 0.01)
final_score = quality_score - time_penalty
```

**Веса метрик:**
- **MRR (40%)** - основная метрика качества ранжирования
- **Precision (20%)** - точность в топ-K результатах
- **Recall (20%)** - полнота в топ-K результатах
- **NDCG (10%)** - нормализованный DCG
- **Hit Rate (10%)** - процент успешных запросов
- **Время поиска** - небольшой штраф за медленность

## ⏱️ Оценка времени выполнения

### Время на одно испытание:
- **50 семплов**: ~10 секунд
- **100 семплов**: ~20 секунд
- **500 семплов**: ~3 минуты

### Общее время оптимизации:
- **5 испытаний**: ~1 минута
- **10 испытаний**: ~2 минуты
- **20 испытаний**: ~5 минут

## 📈 Мониторинг прогресса

### 1. **Консольный вывод**
```
🧪 Запуск теста для испытания 5
📋 Параметры: {'semantic_weight': 0.7, 'bm25_k1': 1.2, ...}
✅ Испытание 5 завершено. Оценка: 0.8234
🏆 Новый лучший результат! Оценка: 0.8234
📊 MRR: 0.8021, Precision: 0.1734, Recall: 0.8640
```

### 2. **MLflow UI**
```bash
# Запуск MLflow UI
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

- Откройте: http://localhost:5000
- Найдите эксперимент: `Simple_Retriever_Optimization`
- Просмотрите метрики: `objective_score`, `mrr`, `precision`, `recall`, `ndcg`, `hit_rate`

### 3. **Файлы результатов**
- `results/simple_hyperparameter_optimization_results.json` - детальные результаты
- `config/best_simple_retriever_config.yaml` - лучшая конфигурация

## 🔍 Анализ результатов

### 1. **Автоматический анализ**
После завершения оптимизации система автоматически выводит:
- Лучший результат и параметры
- Статистику по всем испытаниям
- Средние, медианные значения и стандартное отклонение

### 2. **Ручной анализ через MLflow**
```bash
# Запуск MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Просмотр экспериментов
mlflow experiments list
```

### 3. **Анализ JSON результатов**
```python
import json

with open('results/simple_hyperparameter_optimization_results.json', 'r') as f:
    results = json.load(f)

best_params = results['optimization_info']['best_params']
best_score = results['optimization_info']['best_score']
```

## 🎛️ Настройка оптимизации

### 1. **Изменение диапазонов параметров**
Отредактируйте метод `_create_trial_config()` в `simple_hyperparameter_optimization.py`:

```python
# Пример: более узкий диапазон для semantic_weight
semantic_weight = trial.suggest_float('semantic_weight', 0.6, 0.8, step=0.05)
```

### 2. **Изменение целевой функции**
Модифицируйте метод `_objective()`:

```python
# Пример: больше веса для MRR
quality_score = 0.6 * mrr + 0.15 * precision + 0.15 * recall + 0.05 * ndcg + 0.05 * hit_rate
```

### 3. **Настройка количества семплов**
Измените параметр `max_samples` в методе `_run_single_test()`:

```python
test_result = self._run_single_test(config, max_samples=100)  # Быстрее
test_result = self._run_single_test(config, max_samples=500)  # Точнее
```

## 🚨 Известные проблемы и решения

### 1. **"RAG пайплайн не инициализирован"**
**Проблема:** Пайплайн не инициализируется правильно
**Решение:** Нужно добавить `pipeline.initialize()` после создания

### 2. **JSON сериализация ошибки**
**Проблема:** `TypeError: Object of type int64 is not JSON serializable`
**Решение:** Добавить конвертацию numpy типов в Python типы

### 3. **MLflow эксперименты не видны**
**Проблема:** Эксперименты создаются, но не отображаются в UI
**Решение:** Перезапустить MLflow UI или очистить кэш

## 📋 Рекомендации по использованию

### 1. **Для быстрого тестирования**
- 5-10 испытаний
- 50 семплов на испытание
- Таймаут: 10 минут

### 2. **Для серьезной оптимизации**
- 20-50 испытаний
- 100-500 семплов на испытание
- Таймаут: 1-2 часа

### 3. **Для продакшена**
- 100+ испытаний
- 1000+ семплов на испытание
- Таймаут: 6+ часов

## 🔄 Интеграция с CI/CD

### 1. **Автоматический запуск**
```bash
# В pipeline
python simple_hyperparameter_optimization.py --n-trials 10 --timeout 1800
```

### 2. **Мониторинг результатов**
```python
# Проверка улучшения
best_score = get_best_score_from_mlflow()
if best_score > baseline_score * 1.05:  # 5% улучшение
    deploy_new_config()
```

## 📚 Дополнительные ресурсы

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [TPE Algorithm](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)

## 🎯 Следующие шаги

1. **Исправить инициализацию пайплайна** - добавить `pipeline.initialize()`
2. **Исправить JSON сериализацию** - добавить конвертацию numpy типов
3. **Добавить больше метрик** - расширить целевую функцию
4. **Оптимизировать производительность** - уменьшить время на испытание
5. **Добавить визуализацию** - графики сходимости и важности параметров






