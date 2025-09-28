#!/usr/bin/env python3
"""
Скрипт для сравнения обычного семантического поиска и гибридного поиска.
Позволяет провести A/B тестирование и сравнить метрики.
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.evaluation import RAGEvaluator
from src.dataset_loader import DatasetLoader
from src.pipeline import RAGPipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурацию из YAML файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_comparison_config(base_config: Dict[str, Any], search_type: str) -> Dict[str, Any]:
    """Создает конфигурацию для конкретного типа поиска."""
    config = base_config.copy()
    config['retriever']['search_type'] = search_type
    
    # Обновляем название эксперимента
    if search_type == "hybrid":
        config['mlflow']['experiment_name'] = "RAG_Hybrid_Search_Comparison"
    else:
        config['mlflow']['experiment_name'] = "RAG_Semantic_Search_Comparison"
    
    return config


def run_experiment(config: Dict[str, Any], max_samples: int, search_type: str) -> Dict[str, Any]:
    """Запускает эксперимент с заданной конфигурацией."""
    logger.info(f"🚀 Запуск эксперимента: {search_type} поиск")
    
    # Инициализация MLflow
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=f"{search_type}_search_{int(time.time())}"):
        # Логируем параметры
        mlflow.log_params({
            'search_type': search_type,
            'max_samples': max_samples,
            'semantic_weight': config.get('hybrid_search', {}).get('semantic_weight', 'N/A'),
            'bm25_weight': config.get('hybrid_search', {}).get('bm25_weight', 'N/A'),
            'final_k': config.get('hybrid_search', {}).get('final_k', config['retriever']['k']),
            'embedding_model': config['models']['embedding']['name'],
            'generator_model': config['models']['generator']['name']
        })
        
        # Создание пайплайна
        pipeline = RAGPipeline(config)
        pipeline.initialize()
        
        # Проверяем, что пайплайн инициализирован
        if not hasattr(pipeline, 'retriever') or pipeline.retriever is None:
            logger.error("RAG пайплайн не инициализирован правильно")
            raise RuntimeError("RAG пайплайн не инициализирован")
        
        # Загрузка датасета
        dataset_loader = DatasetLoader(config)
        dataset_path = config['datasets']['sberquad']['path']
        qa_pairs_path = Path(dataset_path) / config['datasets']['sberquad']['qa_pairs_file']
        
        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        # Ограничиваем количество примеров
        if max_samples > 0:
            qa_pairs = qa_pairs[:max_samples]
        
        logger.info(f"Тестирование на {len(qa_pairs)} примерах")
        
        # Оценка пайплайна
        evaluator = RAGEvaluator(config)
        results = evaluator.evaluate_pipeline(pipeline, qa_pairs)
        
        # Логируем метрики
        for metric_name, value in results['metrics'].items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value)
        
        # Сохраняем детальные результаты
        results_file = f"results/{search_type}_search_results.json"
        Path("results").mkdir(exist_ok=True)
        
        # Конвертируем NumPy типы для JSON сериализации
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
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, ensure_ascii=False, indent=2)
        
        mlflow.log_artifact(results_file)
        
        logger.info(f"✅ Эксперимент {search_type} завершен")
        return results


def compare_results(semantic_results: Dict[str, Any], hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
    """Сравнивает результаты двух экспериментов."""
    comparison = {
        'semantic_search': semantic_results['metrics'],
        'hybrid_search': hybrid_results['metrics'],
        'improvements': {}
    }
    
    # Вычисляем улучшения
    for metric_name in semantic_results['metrics']:
        if metric_name in hybrid_results['metrics']:
            semantic_value = semantic_results['metrics'][metric_name]
            hybrid_value = hybrid_results['metrics'][metric_name]
            
            if isinstance(semantic_value, (int, float)) and isinstance(hybrid_value, (int, float)):
                if semantic_value != 0:
                    improvement = ((hybrid_value - semantic_value) / semantic_value) * 100
                else:
                    improvement = 100 if hybrid_value > 0 else 0
                
                comparison['improvements'][metric_name] = {
                    'semantic': semantic_value,
                    'hybrid': hybrid_value,
                    'improvement_percent': improvement,
                    'improvement_absolute': hybrid_value - semantic_value
                }
    
    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """Выводит результаты сравнения в удобном формате."""
    print("\n" + "="*80)
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ: СЕМАНТИЧЕСКИЙ vs ГИБРИДНЫЙ ПОИСК")
    print("="*80)
    
    print(f"\n🔍 Семантический поиск:")
    for metric, value in comparison['semantic_search'].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    print(f"\n🔍 Гибридный поиск:")
    for metric, value in comparison['hybrid_search'].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    print(f"\n📈 УЛУЧШЕНИЯ:")
    print("-" * 80)
    
    # Сортируем метрики по размеру улучшения
    improvements = comparison['improvements']
    sorted_improvements = sorted(
        improvements.items(), 
        key=lambda x: abs(x[1]['improvement_percent']), 
        reverse=True
    )
    
    for metric, data in sorted_improvements:
        improvement = data['improvement_percent']
        absolute = data['improvement_absolute']
        
        if improvement > 0:
            emoji = "📈"
            direction = "улучшение"
        elif improvement < 0:
            emoji = "📉"
            direction = "ухудшение"
        else:
            emoji = "➡️"
            direction = "без изменений"
        
        print(f"  {emoji} {metric}: {improvement:+.1f}% ({absolute:+.4f}) - {direction}")
    
    # Вычисляем общую оценку
    positive_improvements = sum(1 for data in improvements.values() if data['improvement_percent'] > 0)
    total_metrics = len(improvements)
    
    print(f"\n🎯 ОБЩАЯ ОЦЕНКА:")
    print(f"  Улучшено метрик: {positive_improvements}/{total_metrics} ({positive_improvements/total_metrics*100:.1f}%)")
    
    if positive_improvements > total_metrics / 2:
        print("  ✅ Гибридный поиск показывает лучшие результаты!")
    elif positive_improvements < total_metrics / 2:
        print("  ❌ Семантический поиск показывает лучшие результаты")
    else:
        print("  ⚖️ Результаты примерно равны")


def main():
    parser = argparse.ArgumentParser(description="Сравнение семантического и гибридного поиска")
    parser.add_argument("--config", default="config/hybrid_cpu_config.yaml", 
                       help="Путь к конфигурационному файлу")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="Максимальное количество примеров для тестирования")
    parser.add_argument("--semantic-only", action="store_true",
                       help="Запустить только семантический поиск")
    parser.add_argument("--hybrid-only", action="store_true",
                       help="Запустить только гибридный поиск")
    
    args = parser.parse_args()
    
    # Загружаем базовую конфигурацию
    base_config = load_config(args.config)
    
    results = {}
    
    # Запускаем семантический поиск
    if not args.hybrid_only:
        logger.info("🔍 Запуск семантического поиска...")
        semantic_config = create_comparison_config(base_config, "similarity")
        results['semantic'] = run_experiment(semantic_config, args.max_samples, "semantic")
    
    # Запускаем гибридный поиск
    if not args.semantic_only:
        logger.info("🔍 Запуск гибридного поиска...")
        hybrid_config = create_comparison_config(base_config, "hybrid")
        results['hybrid'] = run_experiment(hybrid_config, args.max_samples, "hybrid")
    
    # Сравниваем результаты
    if 'semantic' in results and 'hybrid' in results:
        comparison = compare_results(results['semantic'], results['hybrid'])
        print_comparison(comparison)
        
        # Сохраняем сравнение
        comparison_file = "results/search_comparison.json"
        
        # Конвертируем NumPy типы для JSON сериализации
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
        
        comparison_converted = convert_numpy_types(comparison)
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_converted, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Результаты сравнения сохранены в {comparison_file}")
    
    logger.info("✅ Сравнение завершено!")
    logger.info("📈 Просмотрите результаты в MLflow UI: mlflow ui --backend-store-uri file:./mlruns")


if __name__ == "__main__":
    main()
