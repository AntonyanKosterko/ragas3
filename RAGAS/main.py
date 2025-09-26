"""
Основной скрипт для запуска RAG экспериментов с трекингом в MLflow.
Поддерживает различные конфигурации и автоматическое логирование результатов.
"""

import os
import sys
import yaml
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import RAGPipeline, create_rag_pipeline
from src.evaluation import RAGEvaluator, create_evaluator
from src.data_processing import DataProcessor, create_data_processor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGExperimentRunner:
    """Класс для запуска RAG экспериментов с MLflow."""
    
    def __init__(self, config_path: str):
        """
        Инициализация раннера экспериментов.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.mlflow_config = self.config.get('mlflow', {})
        
        # Настройка MLflow
        self._setup_mlflow()
        
        # Создаем компоненты
        self.pipeline = None
        self.evaluator = None
        self.data_processor = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из YAML файла."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Конфигурация загружена из: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
            raise
    
    def _setup_mlflow(self) -> None:
        """Настраивает MLflow для трекинга экспериментов."""
        # Устанавливаем URI для трекинга
        tracking_uri = self.mlflow_config.get('tracking_uri', 'file:./mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Устанавливаем имя эксперимента
        experiment_name = self.mlflow_config.get('experiment_name', 'RAG_Experiments')
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow настроен: {tracking_uri}, эксперимент: {experiment_name}")
    
    def run_experiment(self, force_rebuild: bool = False, evaluate: bool = True) -> Dict[str, Any]:
        """
        Запускает полный эксперимент RAG системы.
        
        Args:
            force_rebuild: Принудительно пересоздать векторную базу
            evaluate: Выполнять ли оценку системы
            
        Returns:
            Dict[str, Any]: Результаты эксперимента
        """
        logger.info("Начало RAG эксперимента")
        start_time = time.time()
        
        with mlflow.start_run(run_name=self.config['experiment']['name']) as run:
            try:
                # Логируем параметры эксперимента
                self._log_experiment_params()
                
                # Инициализируем компоненты
                self._initialize_components()
                
                # Создаем пайплайн
                self.pipeline = create_rag_pipeline(self.config)
                self.pipeline.initialize(force_rebuild=force_rebuild)
                
                # Логируем информацию о пайплайне
                self._log_pipeline_info()
                
                # Выполняем оценку если нужно
                evaluation_results = None
                if evaluate:
                    evaluation_results = self._run_evaluation()
                    self._log_evaluation_metrics(evaluation_results)
                
                # Тестируем пайплайн на примерах
                test_results = self._run_test_queries()
                self._log_test_metrics(test_results)
                
                # Логируем артефакты
                self._log_artifacts()
                
                # Создаем итоговый отчет
                experiment_results = {
                    'run_id': run.info.run_id,
                    'experiment_name': self.config['experiment']['name'],
                    'pipeline_info': self.pipeline.get_vector_store_info(),
                    'pipeline_stats': self.pipeline.get_stats(),
                    'evaluation_results': evaluation_results,
                    'test_results': test_results,
                    'total_time': time.time() - start_time,
                    'timestamp': time.time()
                }
                
                logger.info(f"Эксперимент завершен успешно за {experiment_results['total_time']:.2f} секунд")
                return experiment_results
                
            except Exception as e:
                logger.error(f"Ошибка в эксперименте: {e}")
                mlflow.log_param("error", str(e))
                raise
    
    def _initialize_components(self) -> None:
        """Инициализирует компоненты системы."""
        self.evaluator = create_evaluator(self.config)
        self.data_processor = create_data_processor(self.config)
        logger.info("Компоненты системы инициализированы")
    
    def _log_experiment_params(self) -> None:
        """Логирует параметры эксперимента в MLflow."""
        # Параметры эксперимента
        mlflow.log_param("experiment_name", self.config['experiment']['name'])
        mlflow.log_param("experiment_description", self.config['experiment']['description'])
        mlflow.log_param("tags", str(self.config['experiment']['tags']))
        
        # Параметры данных
        data_config = self.config['data']
        mlflow.log_param("chunk_size", data_config['chunk_size'])
        mlflow.log_param("chunk_overlap", data_config['chunk_overlap'])
        mlflow.log_param("text_splitter", data_config['text_splitter'])
        
        # Параметры моделей
        models_config = self.config['models']
        mlflow.log_param("embedding_model", models_config['embedding']['name'])
        mlflow.log_param("generator_model", models_config['generator']['name'])
        mlflow.log_param("embedding_device", models_config['embedding']['device'])
        mlflow.log_param("generator_device", models_config['generator']['device'])
        
        # Параметры векторной базы
        vector_config = self.config['vector_store']
        mlflow.log_param("vector_store_type", vector_config['type'])
        mlflow.log_param("collection_name", vector_config['collection_name'])
        
        # Параметры ретривера
        retriever_config = self.config['retriever']
        mlflow.log_param("retriever_k", retriever_config['k'])
        mlflow.log_param("search_type", retriever_config['search_type'])
        
        logger.info("Параметры эксперимента залогированы в MLflow")
    
    def _log_pipeline_info(self) -> None:
        """Логирует информацию о пайплайне."""
        pipeline_info = self.pipeline.get_vector_store_info()
        pipeline_stats = self.pipeline.get_stats()
        
        # Логируем информацию о векторной базе
        if 'document_count' in pipeline_info:
            mlflow.log_metric("document_count", pipeline_info['document_count'])
        
        # Логируем статистику пайплайна
        for stat_name, stat_value in pipeline_stats.items():
            if isinstance(stat_value, (int, float)):
                mlflow.log_metric(f"pipeline_{stat_name}", stat_value)
        
        # Логируем детальные временные метрики
        if 'avg_retrieval_time' in pipeline_stats:
            try:
                mlflow.log_metric("avg_retrieval_time", float(pipeline_stats['avg_retrieval_time']))
                mlflow.log_metric("avg_generation_time", float(pipeline_stats['avg_generation_time']))
                mlflow.log_metric("avg_context_prep_time", float(pipeline_stats['avg_context_prep_time']))
                mlflow.log_metric("median_retrieval_time", float(pipeline_stats['median_retrieval_time']))
                mlflow.log_metric("median_generation_time", float(pipeline_stats['median_generation_time']))
                mlflow.log_metric("median_context_prep_time", float(pipeline_stats['median_context_prep_time']))
                mlflow.log_metric("max_retrieval_time", float(pipeline_stats['max_retrieval_time']))
                mlflow.log_metric("max_generation_time", float(pipeline_stats['max_generation_time']))
                mlflow.log_metric("max_context_prep_time", float(pipeline_stats['max_context_prep_time']))
                mlflow.log_metric("std_retrieval_time", float(pipeline_stats['std_retrieval_time']))
                mlflow.log_metric("std_generation_time", float(pipeline_stats['std_generation_time']))
                mlflow.log_metric("std_context_prep_time", float(pipeline_stats['std_context_prep_time']))
                mlflow.log_metric("retrieval_time_percentage", float(pipeline_stats['retrieval_time_percentage']))
                mlflow.log_metric("generation_time_percentage", float(pipeline_stats['generation_time_percentage']))
                mlflow.log_metric("context_prep_time_percentage", float(pipeline_stats['context_prep_time_percentage']))
                mlflow.log_metric("avg_retrieved_docs", float(pipeline_stats['avg_retrieved_docs']))
                mlflow.log_metric("avg_context_length", float(pipeline_stats['avg_context_length']))
                mlflow.log_metric("avg_answer_length", float(pipeline_stats['avg_answer_length']))
            except Exception as e:
                logger.warning(f"Ошибка при логировании детальных метрик: {e}")
        
        logger.info("Информация о пайплайне залогирована")
    
    def _run_evaluation(self) -> Dict[str, Any]:
        """Запускает оценку системы."""
        logger.info("Запуск оценки системы")
        
        # Загружаем датасет
        dataset_path = self.config['data']['dataset_path']
        if not os.path.exists(dataset_path):
            logger.warning(f"Датасет не найден: {dataset_path}. Создаем пример датасета.")
            self.data_processor.create_sample_dataset(dataset_path, num_samples=50)
        
        qa_dataset = self.data_processor.load_qa_dataset(dataset_path)
        
        # Запускаем оценку
        evaluation_results = self.evaluator.evaluate_pipeline(self.pipeline, qa_dataset)
        
        # Сохраняем результаты
        if self.config.get('evaluation', {}).get('save_predictions', True):
            output_path = f"results/evaluation_{self.config['experiment']['name']}.json"
            self.evaluator.save_results(evaluation_results, output_path)
        
        logger.info("Оценка системы завершена")
        return evaluation_results
    
    def _log_evaluation_metrics(self, evaluation_results: Dict[str, Any]) -> None:
        """Логирует метрики оценки в MLflow."""
        metrics = evaluation_results.get('metrics', {})
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
        
        # Логируем общие метрики
        mlflow.log_metric("total_samples", evaluation_results['total_samples'])
        mlflow.log_metric("evaluation_time", evaluation_results['evaluation_time'])
        
        logger.info("Метрики оценки залогированы в MLflow")
    
    def _run_test_queries(self) -> Dict[str, Any]:
        """Запускает тестовые запросы к системе."""
        logger.info("Запуск тестовых запросов")
        
        test_questions = [
            "Что такое машинное обучение?",
            "Какие типы нейронных сетей существуют?",
            "Как работает алгоритм градиентного спуска?",
            "Что такое глубокое обучение?",
            "Какие метрики используются для оценки моделей?"
        ]
        
        test_results = {
            'questions': test_questions,
            'responses': [],
            'avg_response_time': 0,
            'total_time': 0
        }
        
        start_time = time.time()
        response_times = []
        
        for question in test_questions:
            response = self.pipeline.query(question)
            test_results['responses'].append(response)
            response_times.append(response['response_time'])
        
        test_results['total_time'] = time.time() - start_time
        test_results['avg_response_time'] = sum(response_times) / len(response_times)
        
        logger.info("Тестовые запросы завершены")
        return test_results
    
    def _log_test_metrics(self, test_results: Dict[str, Any]) -> None:
        """Логирует метрики тестирования в MLflow."""
        mlflow.log_metric("test_avg_response_time", test_results['avg_response_time'])
        mlflow.log_metric("test_total_time", test_results['total_time'])
        mlflow.log_metric("test_questions_count", len(test_results['questions']))
        
        logger.info("Метрики тестирования залогированы в MLflow")
    
    def _log_artifacts(self) -> None:
        """Логирует артефакты в MLflow."""
        # Логируем конфигурацию
        mlflow.log_artifact(self.config_path)
        
        # Логируем логи
        if os.path.exists('logs/rag_experiment.log'):
            mlflow.log_artifact('logs/rag_experiment.log')
        
        # Логируем результаты оценки если есть
        results_dir = "results"
        if os.path.exists(results_dir):
            mlflow.log_artifacts(results_dir)
        
        logger.info("Артефакты залогированы в MLflow")
    
    def compare_experiments(self, run_ids: list) -> Dict[str, Any]:
        """
        Сравнивает результаты нескольких экспериментов.
        
        Args:
            run_ids: Список ID запусков для сравнения
            
        Returns:
            Dict[str, Any]: Результаты сравнения
        """
        logger.info(f"Сравнение экспериментов: {run_ids}")
        
        client = MlflowClient()
        comparison_results = {
            'run_ids': run_ids,
            'metrics_comparison': {},
            'best_runs': {}
        }
        
        # Получаем метрики для каждого запуска
        all_metrics = set()
        run_metrics = {}
        
        for run_id in run_ids:
            try:
                run = client.get_run(run_id)
                metrics = run.data.metrics
                run_metrics[run_id] = metrics
                all_metrics.update(metrics.keys())
            except Exception as e:
                logger.warning(f"Ошибка при получении метрик для {run_id}: {e}")
        
        # Сравниваем метрики
        for metric in all_metrics:
            metric_values = {}
            for run_id, metrics in run_metrics.items():
                if metric in metrics:
                    metric_values[run_id] = metrics[metric]
            
            if metric_values:
                comparison_results['metrics_comparison'][metric] = metric_values
                
                # Находим лучший результат
                best_run_id = max(metric_values.items(), key=lambda x: x[1])[0]
                comparison_results['best_runs'][metric] = best_run_id
        
        return comparison_results


def main():
    """Основная функция для запуска скрипта."""
    parser = argparse.ArgumentParser(description='RAG Experiment Runner')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Принудительно пересоздать векторную базу')
    parser.add_argument('--no-evaluation', action='store_true',
                       help='Пропустить оценку системы')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Сравнить результаты указанных запусков')
    
    args = parser.parse_args()
    
    # Создаем необходимые директории
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    try:
        # Создаем раннер экспериментов
        runner = RAGExperimentRunner(args.config)
        
        if args.compare:
            # Режим сравнения
            comparison_results = runner.compare_experiments(args.compare)
            print("\nРезультаты сравнения экспериментов:")
            print("=" * 50)
            for metric, values in comparison_results['metrics_comparison'].items():
                print(f"\n{metric}:")
                for run_id, value in values.items():
                    print(f"  {run_id}: {value:.4f}")
        else:
            # Режим запуска эксперимента
            results = runner.run_experiment(
                force_rebuild=args.force_rebuild,
                evaluate=not args.no_evaluation
            )
            
            print(f"\nЭксперимент завершен успешно!")
            print(f"Run ID: {results['run_id']}")
            print(f"Время выполнения: {results['total_time']:.2f} секунд")
            
            if results['evaluation_results']:
                metrics = results['evaluation_results']['metrics']
                print(f"\nОсновные метрики:")
                for metric, value in metrics.items():
                    if metric.endswith('_mean'):
                        print(f"  {metric}: {value:.4f}")
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении скрипта: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

