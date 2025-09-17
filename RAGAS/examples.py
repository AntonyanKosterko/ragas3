"""
Примеры использования RAG системы.
Демонстрирует различные сценарии применения и настройки.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import create_rag_pipeline, RAGPipelineManager
from src.evaluation import create_evaluator
from src.data_processing import create_data_processor
from src.models import create_model_manager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Пример базового использования RAG системы."""
    print("=" * 60)
    print("ПРИМЕР 1: Базовое использование RAG системы")
    print("=" * 60)
    
    # Загружаем конфигурацию
    with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Создаем и инициализируем пайплайн
    print("Создание RAG пайплайна...")
    pipeline = create_rag_pipeline(config)
    pipeline.initialize()
    
    # Задаем вопросы
    questions = [
        "Что такое машинное обучение?",
        "Какие типы нейронных сетей существуют?",
        "Как работает градиентный спуск?"
    ]
    
    print("\nТестирование пайплайна:")
    for question in questions:
        print(f"\nВопрос: {question}")
        result = pipeline.query(question)
        print(f"Ответ: {result['answer'][:200]}...")
        print(f"Время ответа: {result['response_time']:.2f} сек")
    
    # Получаем статистику
    stats = pipeline.get_stats()
    print(f"\nСтатистика пайплайна:")
    for stat_name, stat_value in stats.items():
        print(f"  {stat_name}: {stat_value}")


def example_evaluation():
    """Пример оценки качества системы."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 2: Оценка качества RAG системы")
    print("=" * 60)
    
    # Загружаем конфигурацию
    with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Создаем компоненты
    pipeline = create_rag_pipeline(config)
    pipeline.initialize()
    
    evaluator = create_evaluator(config)
    data_processor = create_data_processor(config)
    
    # Загружаем датасет
    qa_dataset = data_processor.load_qa_dataset('data/russian_qa_dataset.json')
    print(f"Загружено {len(qa_dataset)} пар вопрос-ответ")
    
    # Запускаем оценку
    print("Запуск оценки...")
    evaluation_results = evaluator.evaluate_pipeline(pipeline, qa_dataset)
    
    # Выводим результаты
    print("\nРезультаты оценки:")
    for metric, value in evaluation_results['metrics'].items():
        if metric.endswith('_mean'):
            print(f"  {metric}: {value:.4f}")
    
    # Сохраняем результаты
    evaluator.save_results(evaluation_results, 'results/example_evaluation.json')
    print("\nРезультаты сохранены в results/example_evaluation.json")


def example_multiple_pipelines():
    """Пример работы с несколькими пайплайнами."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 3: Работа с несколькими пайплайнами")
    print("=" * 60)
    
    # Создаем менеджер пайплайнов
    manager = RAGPipelineManager()
    
    # Загружаем конфигурации
    configs = {
        'base': 'config/base_config.yaml',
        'gpu': 'config/gpu_config.yaml',
        'cpu': 'config/cpu_config.yaml'
    }
    
    # Создаем пайплайны
    for name, config_path in configs.items():
        if os.path.exists(config_path):
            print(f"Создание пайплайна: {name}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            manager.add_pipeline(name, config)
        else:
            print(f"Конфигурация не найдена: {config_path}")
    
    # Устанавливаем активный пайплайн
    manager.set_active_pipeline('base')
    active_pipeline = manager.get_active_pipeline()
    
    if active_pipeline:
        print(f"\nАктивный пайплайн: base")
        result = active_pipeline.query("Что такое глубокое обучение?")
        print(f"Ответ: {result['answer'][:200]}...")
    
    # Список всех пайплайнов
    print(f"\nДоступные пайплайны: {manager.list_pipelines()}")


def example_custom_configuration():
    """Пример создания кастомной конфигурации."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 4: Кастомная конфигурация")
    print("=" * 60)
    
    # Создаем кастомную конфигурацию
    custom_config = {
        'experiment': {
            'name': 'custom_experiment',
            'description': 'Кастомный эксперимент с особыми настройками',
            'tags': ['custom', 'example']
        },
        'data': {
            'input_path': 'data/documents',
            'dataset_path': 'data/russian_qa_dataset.json',
            'chunk_size': 500,
            'chunk_overlap': 100,
            'text_splitter': 'recursive'
        },
        'models': {
            'embedding': {
                'name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'device': 'auto',
                'normalize_embeddings': True
            },
            'generator': {
                'name': 'microsoft/DialoGPT-small',
                'device': 'auto',
                'max_length': 256,
                'temperature': 0.8,
                'do_sample': True
            }
        },
        'vector_store': {
            'type': 'chroma',
            'persist_directory': 'data/vector_db_custom',
            'collection_name': 'custom_documents'
        },
        'retriever': {
            'k': 3,
            'search_type': 'similarity',
            'fetch_k': 10
        },
        'evaluation': {
            'metrics': ['cosine_similarity', 'rouge'],
            'batch_size': 4,
            'save_predictions': True
        },
        'mlflow': {
            'experiment_name': 'Custom_Experiments',
            'tracking_uri': 'file:./mlruns',
            'log_artifacts': True,
            'log_models': True
        },
        'web': {
            'interface': 'gradio',
            'port': 7860,
            'host': '0.0.0.0',
            'share': False
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/custom_experiment.log'
        }
    }
    
    # Сохраняем конфигурацию
    config_path = 'config/custom_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(custom_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Кастомная конфигурация сохранена в: {config_path}")
    
    # Создаем пайплайн с кастомной конфигурацией
    pipeline = create_rag_pipeline(custom_config)
    pipeline.initialize()
    
    # Тестируем
    result = pipeline.query("Что такое искусственный интеллект?")
    print(f"Ответ: {result['answer'][:200]}...")


def example_model_comparison():
    """Пример сравнения разных моделей."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 5: Сравнение разных моделей")
    print("=" * 60)
    
    # Список моделей для сравнения
    embedding_models = [
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'sentence-transformers/distiluse-base-multilingual-cased'
    ]
    
    generator_models = [
        'microsoft/DialoGPT-small',
        'microsoft/DialoGPT-medium'
    ]
    
    # Загружаем базовую конфигурацию
    with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    test_question = "Что такое машинное обучение?"
    
    print(f"Тестовый вопрос: {test_question}")
    print("\nСравнение моделей:")
    
    for emb_model in embedding_models:
        for gen_model in generator_models:
            print(f"\n--- Embedding: {emb_model.split('/')[-1]} | Generator: {gen_model.split('/')[-1]} ---")
            
            # Создаем конфигурацию с новыми моделями
            config = base_config.copy()
            config['models']['embedding']['name'] = emb_model
            config['models']['generator']['name'] = gen_model
            config['experiment']['name'] = f"comparison_{emb_model.split('/')[-1]}_{gen_model.split('/')[-1]}"
            
            try:
                # Создаем и тестируем пайплайн
                pipeline = create_rag_pipeline(config)
                pipeline.initialize()
                
                result = pipeline.query(test_question)
                print(f"Ответ: {result['answer'][:150]}...")
                print(f"Время ответа: {result['response_time']:.2f} сек")
                
            except Exception as e:
                print(f"Ошибка: {str(e)[:100]}...")


def example_batch_processing():
    """Пример пакетной обработки вопросов."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 6: Пакетная обработка вопросов")
    print("=" * 60)
    
    # Загружаем конфигурацию
    with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Создаем пайплайн
    pipeline = create_rag_pipeline(config)
    pipeline.initialize()
    
    # Список вопросов для пакетной обработки
    batch_questions = [
        "Что такое машинное обучение?",
        "Какие типы нейронных сетей существуют?",
        "Как работает градиентный спуск?",
        "Что такое переобучение?",
        "Какие метрики используются для оценки моделей?",
        "Что такое регуляризация?",
        "Как работает кросс-валидация?",
        "Что такое глубокое обучение?",
        "Какие алгоритмы используются в ML?",
        "Что такое компьютерное зрение?"
    ]
    
    print(f"Обработка {len(batch_questions)} вопросов...")
    
    # Пакетная обработка
    results = pipeline.batch_query(batch_questions)
    
    # Анализ результатов
    response_times = [r['response_time'] for r in results]
    avg_time = sum(response_times) / len(response_times)
    total_time = sum(response_times)
    
    print(f"\nРезультаты пакетной обработки:")
    print(f"  Всего вопросов: {len(batch_questions)}")
    print(f"  Среднее время ответа: {avg_time:.2f} сек")
    print(f"  Общее время: {total_time:.2f} сек")
    print(f"  Вопросов в секунду: {len(batch_questions) / total_time:.2f}")
    
    # Показываем несколько примеров
    print(f"\nПримеры ответов:")
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. Вопрос: {result['question']}")
        print(f"   Ответ: {result['answer'][:100]}...")
        print(f"   Время: {result['response_time']:.2f} сек")


def main():
    """Запуск всех примеров."""
    print("🚀 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ RAG СИСТЕМЫ")
    print("=" * 60)
    
    try:
        # Создаем необходимые директории
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Запускаем примеры
        example_basic_usage()
        example_evaluation()
        example_multiple_pipelines()
        example_custom_configuration()
        example_model_comparison()
        example_batch_processing()
        
        print("\n" + "=" * 60)
        print("✅ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении примеров: {e}")
        logger.exception("Детали ошибки:")


if __name__ == "__main__":
    main()

