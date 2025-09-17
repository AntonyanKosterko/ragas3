"""
Модуль для тестирования RAG системы на датасетах с метриками
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class RAGTester:
    """Класс для тестирования RAG системы на датасетах"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация тестера RAG.
        
        Args:
            config: Конфигурация из YAML файла
        """
        self.config = config
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_scorer = BLEU()
        
    def load_qa_pairs(self, qa_pairs_path: str) -> List[Dict[str, Any]]:
        """
        Загружает пары вопрос-ответ из файла.
        
        Args:
            qa_pairs_path: Путь к файлу с парами вопрос-ответ
            
        Returns:
            List[Dict[str, Any]]: Список пар вопрос-ответ
        """
        logger.info(f"Загрузка пар вопрос-ответ из {qa_pairs_path}")
        
        if not os.path.exists(qa_pairs_path):
            raise FileNotFoundError(f"Файл {qa_pairs_path} не найден")
        
        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        logger.info(f"Загружено {len(qa_pairs)} пар вопрос-ответ")
        return qa_pairs
    
    def test_rag_system(self, pipeline, qa_pairs: List[Dict[str, Any]], 
                       max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Тестирует RAG систему на парах вопрос-ответ.
        
        Args:
            pipeline: RAG пайплайн
            qa_pairs: Список пар вопрос-ответ
            max_samples: Максимальное количество примеров для тестирования
            
        Returns:
            Dict[str, Any]: Результаты тестирования
        """
        logger.info(f"Начало тестирования RAG системы на {len(qa_pairs)} примерах")
        
        if max_samples:
            qa_pairs = qa_pairs[:max_samples]
            logger.info(f"Ограничено до {max_samples} примеров")
        
        results = {
            'total_samples': len(qa_pairs),
            'test_results': [],
            'metrics': {},
            'response_times': [],
            'start_time': time.time()
        }
        
        for i, qa_pair in enumerate(qa_pairs):
            try:
                question = qa_pair['question']
                expected_answer = qa_pair.get('answer', '')
                
                # Выполняем запрос к RAG системе
                start_time = time.time()
                rag_result = pipeline.query(question, return_sources=True)
                response_time = time.time() - start_time
                
                predicted_answer = rag_result.get('answer', '')
                source_documents = rag_result.get('source_documents', [])
                
                # Сохраняем результат
                test_result = {
                    'question_id': qa_pair.get('id', i),
                    'question': question,
                    'expected_answer': expected_answer,
                    'predicted_answer': predicted_answer,
                    'response_time': response_time,
                    'source_documents_count': len(source_documents),
                    'source_documents': source_documents
                }
                
                results['test_results'].append(test_result)
                results['response_times'].append(response_time)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Обработано {i + 1}/{len(qa_pairs)} примеров")
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке примера {i}: {e}")
                continue
        
        # Вычисляем метрики
        results['metrics'] = self._calculate_metrics(results['test_results'])
        results['total_time'] = time.time() - results['start_time']
        results['avg_response_time'] = np.mean(results['response_times'])
        
        logger.info(f"Тестирование завершено за {results['total_time']:.2f} секунд")
        return results
    
    def _calculate_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Вычисляет метрики качества RAG системы.
        
        Args:
            test_results: Результаты тестирования
            
        Returns:
            Dict[str, float]: Метрики качества
        """
        logger.info("Вычисление метрик качества")
        
        metrics = {}
        
        # Метрики на основе ответов
        predicted_answers = [result['predicted_answer'] for result in test_results]
        expected_answers = [result['expected_answer'] for result in test_results]
        
        # ROUGE метрики
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, exp in zip(predicted_answers, expected_answers):
            if pred and exp:
                scores = self.rouge_scorer.score(exp, pred)
                for metric in rouge_scores.keys():
                    rouge_scores[metric].append(scores[metric].fmeasure)
        
        for metric in rouge_scores.keys():
            if rouge_scores[metric]:
                metrics[f'{metric}_mean'] = np.mean(rouge_scores[metric])
                metrics[f'{metric}_std'] = np.std(rouge_scores[metric])
        
        # BLEU метрика
        if predicted_answers and expected_answers:
            # Фильтруем пустые ответы
            valid_pairs = [(pred, exp) for pred, exp in zip(predicted_answers, expected_answers) 
                          if pred and exp]
            
            if valid_pairs:
                preds, exps = zip(*valid_pairs)
                bleu_score = self.bleu_scorer.corpus_score(preds, [exps])
                metrics['bleu'] = bleu_score.score
        
        # Exact Match
        exact_matches = 0
        for pred, exp in zip(predicted_answers, expected_answers):
            if pred and exp and pred.strip().lower() == exp.strip().lower():
                exact_matches += 1
        
        metrics['exact_match'] = exact_matches / len(test_results) if test_results else 0
        
        # Length Ratio
        length_ratios = []
        for pred, exp in zip(predicted_answers, expected_answers):
            if pred and exp:
                ratio = len(pred) / len(exp) if len(exp) > 0 else 0
                length_ratios.append(ratio)
        
        if length_ratios:
            metrics['length_ratio_mean'] = np.mean(length_ratios)
            metrics['length_ratio_std'] = np.std(length_ratios)
        
        # Response Time метрики
        response_times = [result['response_time'] for result in test_results]
        if response_times:
            metrics['response_time_mean'] = np.mean(response_times)
            metrics['response_time_std'] = np.std(response_times)
            metrics['response_time_min'] = np.min(response_times)
            metrics['response_time_max'] = np.max(response_times)
        
        # Source Documents метрики
        source_counts = [result['source_documents_count'] for result in test_results]
        if source_counts:
            metrics['avg_sources_per_query'] = np.mean(source_counts)
            metrics['queries_with_sources'] = sum(1 for count in source_counts if count > 0) / len(source_counts)
        
        logger.info(f"Вычислено {len(metrics)} метрик")
        return metrics
    
    def log_results_to_mlflow(self, results: Dict[str, Any], experiment_name: str = "RAG_Dataset_Testing"):
        """
        Логирует результаты тестирования в MLflow.
        
        Args:
            results: Результаты тестирования
            experiment_name: Имя эксперимента в MLflow
        """
        logger.info(f"Логирование результатов в MLflow: {experiment_name}")
        
        # Устанавливаем эксперимент
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"rag_test_{int(time.time())}"):
            # Логируем параметры
            mlflow.log_param("total_samples", results['total_samples'])
            mlflow.log_param("test_type", "rag_dataset_testing")
            
            # Логируем метрики
            for metric_name, metric_value in results['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
            
            # Логируем общие метрики
            mlflow.log_metric("total_time", results['total_time'])
            mlflow.log_metric("avg_response_time", results['avg_response_time'])
            
            # Сохраняем детальные результаты
            results_file = f"rag_test_results_{int(time.time())}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            mlflow.log_artifact(results_file)
            
            # Удаляем временный файл
            os.remove(results_file)
        
        logger.info("Результаты успешно залогированы в MLflow")
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Сохраняет результаты тестирования в файл.
        
        Args:
            results: Результаты тестирования
            output_path: Путь для сохранения результатов
        """
        logger.info(f"Сохранение результатов в {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("Результаты сохранены")


def create_rag_tester(config: Dict[str, Any]) -> RAGTester:
    """Создает экземпляр RAGTester"""
    return RAGTester(config)
