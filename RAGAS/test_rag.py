#!/usr/bin/env python3
"""
Единый тестер RAG системы с датасетом SberQuAD
"""

import os
import sys
import yaml
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import create_rag_pipeline
from src.dataset_loader import create_dataset_loader
from src.evaluation import RAGEvaluator

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rag_system(config_path: str, max_samples: int = None, rebuild_vector_db: bool = False):
    """Тестирует RAG систему на датасете SberQuAD"""
    logger.info("🚀 Запуск тестирования RAG системы с датасетом SberQuAD")
    
    # Загружаем конфигурацию
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Создаем компоненты
    dataset_loader = create_dataset_loader(config)
    evaluator = RAGEvaluator(config)
    
    # Получаем информацию о датасете
    datasets_config = config.get('datasets', {})
    if not datasets_config:
        logger.error("В конфигурации не найдены настройки датасетов")
        return
    
    dataset_name = list(datasets_config.keys())[0]
    dataset_config = datasets_config[dataset_name]
    dataset_path = dataset_config['path']
    
    # Проверяем, существует ли датасет
    if not os.path.exists(dataset_path):
        logger.error(f"Датасет не найден: {dataset_path}")
        logger.info("Запустите сначала: python load_sberquad.py")
        return
    
    dataset_info = dataset_loader.get_dataset_info(dataset_path)
    logger.info(f"📊 Информация о датасете: {dataset_info}")
    
    if dataset_info['documents_count'] == 0:
        logger.error("В датасете не найдено документов")
        return
    
    # Создаем или загружаем векторную БД
    vector_db_path = dataset_config['vector_db_path']
    
    if rebuild_vector_db or not os.path.exists(vector_db_path):
        logger.info("Создание векторной БД из датасета...")
        dataset_loader.create_vector_store_from_dataset(dataset_path, vector_db_path)
    else:
        logger.info("Векторная БД уже существует, пропускаем создание")
    
    # Создаем RAG пайплайн
    logger.info("Создание RAG пайплайна...")
    pipeline = create_rag_pipeline(config)
    pipeline.initialize()
    
    # Загружаем векторную БД из датасета
    logger.info("Загрузка векторной БД...")
    vector_store = dataset_loader.load_vector_store_from_dataset(dataset_path, vector_db_path)
    
    # Создаем ретривер через DataProcessor для поддержки гибридного поиска
    from src.data_processing import DataProcessor
    data_processor = DataProcessor(config)
    data_processor.vector_store = vector_store
    retriever = data_processor.create_retriever(config['retriever']['search_type'])
    
    # Обновляем пайплайн с новой векторной БД
    pipeline.vector_store = vector_store
    pipeline.retriever = retriever
    pipeline._create_qa_chain()
    
    logger.info("RAG пайплайн готов для тестирования")
    
    # Загружаем пары вопрос-ответ для тестирования
    qa_pairs_path = os.path.join(dataset_path, dataset_config['qa_pairs_file'])
    if os.path.exists(qa_pairs_path):
        import json
        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        logger.info(f"Загружено {len(qa_pairs)} пар вопрос-ответ")
    else:
        logger.error("Файл с парами вопрос-ответ не найден")
        return
    
    # Ограничиваем количество примеров если нужно
    if max_samples and max_samples < len(qa_pairs):
        qa_pairs = qa_pairs[:max_samples]
        logger.info(f"Ограничено до {max_samples} примеров")
    
    logger.info(f"Начало тестирования на {len(qa_pairs)} примерах")
    
    # Запускаем тестирование с новым оценщиком
    results = evaluator.evaluate_pipeline(pipeline, qa_pairs)
    
    # Выводим результаты
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ RAG СИСТЕМЫ")
    print("="*60)
    print(f"Всего примеров: {results['total_samples']}")
    print(f"Время выполнения: {results['evaluation_time']:.2f} сек")
    
    # Вычисляем среднее время ответа
    response_times = [pred['metrics'].get('response_time', 0) for pred in results['predictions']]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    print(f"Среднее время ответа: {avg_response_time:.3f} сек")
    
    print("\nМетрики качества:")
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    # Логируем в MLflow
    try:
        import mlflow
        import mlflow.sklearn
        
        mlflow_config = config.get('mlflow', {})
        experiment_name = mlflow_config.get('experiment_name', 'RAG_SberQuAD_Testing')
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Логируем параметры
            mlflow.log_params({
                'total_queries': results['total_samples'],
                'dataset_name': dataset_name,
                'config_path': config_path,
                'max_samples': max_samples or results['total_samples']
            })
            
            # Логируем метрики
            for metric, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric, value)
            
            # Логируем артефакты
            if os.path.exists("results/rag_test_results.json"):
                mlflow.log_artifact("results/rag_test_results.json")
        
        logger.info("Результаты успешно залогированы в MLflow")
        
    except Exception as e:
        logger.error(f"Ошибка при логировании в MLflow: {e}")
    
    # Сохраняем результаты
    os.makedirs("results", exist_ok=True)
    import json
    import numpy as np
    
    # Функция для конвертации numpy типов
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_converted = convert_numpy_types(results)
    results_file = "results/rag_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Результаты сохранены в {results_file}")
    logger.info("✅ Тестирование завершено!")
    
    return results


def main():
    """Основная функция для тестирования RAG"""
    parser = argparse.ArgumentParser(description='RAG System Testing with SberQuAD')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Максимальное количество примеров для тестирования')
    parser.add_argument('--rebuild-vector-db', action='store_true',
                       help='Пересоздать векторную БД из датасета')
    
    args = parser.parse_args()
    
    try:
        results = test_rag_system(args.config, args.max_samples, args.rebuild_vector_db)
        
        if results:
            print(f"\n✅ Тестирование завершено успешно!")
            print(f"📊 Результаты сохранены в results/rag_test_results.json")
            print(f"📈 Просмотрите результаты в MLflow UI: mlflow ui --backend-store-uri file:./mlruns")
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        raise


if __name__ == "__main__":
    main()






