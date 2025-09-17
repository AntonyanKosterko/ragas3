"""
Скрипт для тестирования RAG системы на датасете RAG Bench
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import create_rag_pipeline
from src.dataset_loader import create_dataset_loader
from src.rag_tester import create_rag_tester

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Основная функция для тестирования RAG на датасете"""
    parser = argparse.ArgumentParser(description='RAG Dataset Testing')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--dataset', type=str, default='datasets/rag_bench',
                       help='Путь к датасету')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Максимальное количество примеров для тестирования')
    parser.add_argument('--rebuild-vector-db', action='store_true',
                       help='Пересоздать векторную БД из датасета')
    parser.add_argument('--output', type=str, default='results/rag_dataset_test_results.json',
                       help='Путь для сохранения результатов')
    
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Конфигурация загружена из {args.config}")
        
        # Создаем компоненты
        dataset_loader = create_dataset_loader(config)
        rag_tester = create_rag_tester(config)
        
        # Получаем информацию о датасете
        dataset_info = dataset_loader.get_dataset_info(args.dataset)
        logger.info(f"Информация о датасете: {dataset_info}")
        
        if dataset_info['documents_count'] == 0:
            logger.error("В датасете не найдено документов")
            return
        
        # Создаем или загружаем векторную БД
        vector_db_path = f"{args.dataset}/vector_db"
        
        if args.rebuild_vector_db or not os.path.exists(vector_db_path):
            logger.info("Создание векторной БД из датасета...")
            dataset_loader.create_vector_store_from_dataset(args.dataset, vector_db_path)
        else:
            logger.info("Векторная БД уже существует, пропускаем создание")
        
        # Создаем RAG пайплайн
        logger.info("Создание RAG пайплайна...")
        pipeline = create_rag_pipeline(config)
        pipeline.initialize()
        
        # Загружаем векторную БД из датасета
        logger.info("Загрузка векторной БД...")
        vector_store = dataset_loader.load_vector_store_from_dataset(args.dataset, vector_db_path)
        
        # Создаем ретривер
        from langchain_core.retrievers import BaseRetriever
        retriever = vector_store.as_retriever(
            search_type=config['retriever']['search_type'],
            search_kwargs={'k': config['retriever']['k']}
        )
        
        # Обновляем пайплайн с новой векторной БД
        pipeline.vector_store = vector_store
        pipeline.retriever = retriever
        pipeline._create_qa_chain()
        
        logger.info("RAG пайплайн готов для тестирования")
        
        # Загружаем пары вопрос-ответ для тестирования
        qa_pairs_path = os.path.join(args.dataset, "qa_pairs.json")
        if os.path.exists(qa_pairs_path):
            qa_pairs = rag_tester.load_qa_pairs(qa_pairs_path)
        else:
            logger.warning("Файл с парами вопрос-ответ не найден, создаем тестовые вопросы")
            # Создаем тестовые вопросы на основе документов
            qa_pairs = create_test_questions_from_documents(args.dataset)
        
        if not qa_pairs:
            logger.error("Нет пар вопрос-ответ для тестирования")
            return
        
        # Тестируем RAG систему
        logger.info(f"Начало тестирования на {len(qa_pairs)} примерах")
        results = rag_tester.test_rag_system(pipeline, qa_pairs, args.max_samples)
        
        # Выводим результаты
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ RAG НА ДАТАСЕТЕ")
        print("="*60)
        print(f"Всего примеров: {results['total_samples']}")
        print(f"Время выполнения: {results['total_time']:.2f} сек")
        print(f"Среднее время ответа: {results['avg_response_time']:.2f} сек")
        print("\nМетрики качества:")
        for metric, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
        
        # Логируем в MLflow
        rag_tester.log_results_to_mlflow(results, "RAG_Dataset_Testing")
        
        # Сохраняем результаты
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        rag_tester.save_results(results, args.output)
        
        logger.info(f"Тестирование завершено. Результаты сохранены в {args.output}")
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        raise


def create_test_questions_from_documents(dataset_path: str) -> list:
    """Создает тестовые вопросы на основе документов из датасета"""
    logger.info("Создание тестовых вопросов на основе документов")
    
    documents_file = os.path.join(dataset_path, "documents_for_rag.json")
    if not os.path.exists(documents_file):
        return []
    
    import json
    with open(documents_file, 'r', encoding='utf-8') as f:
        documents_data = json.load(f)
    
    # Простые тестовые вопросы
    test_questions = [
        "Что такое машинное обучение?",
        "Какие типы нейронных сетей существуют?",
        "Как работает алгоритм градиентного спуска?",
        "Что такое глубокое обучение?",
        "Какие метрики используются для оценки моделей?",
        "Что такое переобучение в машинном обучении?",
        "Какие алгоритмы оптимизации используются?",
        "Что такое регуляризация?",
        "Какие проблемы возникают в машинном обучении?",
        "Как оценивается качество моделей?"
    ]
    
    qa_pairs = []
    for i, question in enumerate(test_questions):
        qa_pairs.append({
            'id': f'test_question_{i}',
            'question': question,
            'answer': '',  # Пустой ответ для тестирования
            'source': 'test',
            'metadata': {'type': 'test_question'}
        })
    
    # Сохраняем тестовые вопросы
    qa_pairs_file = os.path.join(dataset_path, "qa_pairs.json")
    with open(qa_pairs_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Создано {len(qa_pairs)} тестовых вопросов")
    return qa_pairs


if __name__ == "__main__":
    main()
