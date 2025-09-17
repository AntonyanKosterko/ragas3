"""
Модуль для оценки качества RAG системы.
Включает различные метрики: косинусное сходство, ROUGE, BLEU и другие.
"""

import logging
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Загружаем необходимые ресурсы NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


class RAGEvaluator:
    """Класс для оценки качества RAG системы."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация оценщика.
        
        Args:
            config: Конфигурация из YAML файла
        """
        self.config = config
        self.evaluation_config = config.get('evaluation', {})
        self.metrics = self.evaluation_config.get('metrics', ['cosine_similarity'])
        
        # Инициализируем модели для оценки
        self.embedding_model = None
        self.rouge_scorer = None
        self.bleu_scorer = None
        
        self._initialize_evaluation_models()
    
    def _initialize_evaluation_models(self) -> None:
        """Инициализирует модели для вычисления метрик."""
        try:
            # Модель для семантического сходства
            if 'cosine_similarity' in self.metrics:
                self.embedding_model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                )
                logger.info("Модель для семантического сходства загружена")
            
            # ROUGE scorer
            if 'rouge' in self.metrics:
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=True
                )
                logger.info("ROUGE scorer инициализирован")
            
            # BLEU scorer
            if 'bleu' in self.metrics:
                self.bleu_scorer = BLEU()
                logger.info("BLEU scorer инициализирован")
                
        except Exception as e:
            logger.error(f"Ошибка при инициализации моделей оценки: {e}")
    
    def evaluate_pipeline(self, pipeline: RAGPipeline, qa_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Оценивает качество RAG пайплайна на датасете.
        
        Args:
            pipeline: RAG пайплайн для оценки
            qa_dataset: Датасет вопросов и ответов
            
        Returns:
            Dict[str, Any]: Результаты оценки
        """
        logger.info(f"Начало оценки пайплайна на {len(qa_dataset)} примерах")
        
        results = {
            'total_samples': len(qa_dataset),
            'metrics': {},
            'predictions': [],
            'evaluation_time': 0,
            'timestamp': time.time()
        }
        
        start_time = time.time()
        
        # Обрабатываем каждый пример
        for i, sample in enumerate(qa_dataset):
            if i % 10 == 0:
                logger.info(f"Обработка примера {i+1}/{len(qa_dataset)}")
            
            question = sample['question']
            ground_truth = sample['answer']
            
            # Получаем ответ от пайплайна
            try:
                response = pipeline.query(question, return_sources=True)
                predicted_answer = response['answer']
                response_time = response['response_time']
            except Exception as e:
                logger.warning(f"Ошибка при обработке вопроса {i+1}: {e}")
                predicted_answer = ""
                response_time = 0
            
            # Вычисляем метрики
            sample_metrics = self._compute_metrics(predicted_answer, ground_truth)
            sample_metrics['response_time'] = response_time
            
            # Сохраняем предсказание
            prediction = {
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted_answer,
                'metrics': sample_metrics,
                'sample_id': i
            }
            
            results['predictions'].append(prediction)
        
        # Вычисляем агрегированные метрики
        results['metrics'] = self._aggregate_metrics(results['predictions'])
        results['evaluation_time'] = time.time() - start_time
        
        logger.info(f"Оценка завершена за {results['evaluation_time']:.2f} секунд")
        return results
    
    def _compute_metrics(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """
        Вычисляет метрики для одного примера.
        
        Args:
            predicted: Предсказанный ответ
            ground_truth: Эталонный ответ
            
        Returns:
            Dict[str, float]: Словарь с метриками
        """
        metrics = {}
        
        # Косинусное сходство
        if 'cosine_similarity' in self.metrics and self.embedding_model:
            try:
                embeddings = self.embedding_model.encode([predicted, ground_truth])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                metrics['cosine_similarity'] = float(similarity)
            except Exception as e:
                logger.warning(f"Ошибка при вычислении косинусного сходства: {e}")
                metrics['cosine_similarity'] = 0.0
        
        # ROUGE метрики
        if 'rouge' in self.metrics and self.rouge_scorer:
            try:
                rouge_scores = self.rouge_scorer.score(ground_truth, predicted)
                metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
                metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
            except Exception as e:
                logger.warning(f"Ошибка при вычислении ROUGE: {e}")
                metrics.update({'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0})
        
        # BLEU метрика
        if 'bleu' in self.metrics and self.bleu_scorer:
            try:
                # Используем NLTK BLEU для более точного вычисления
                smoothie = SmoothingFunction().method4
                bleu_score = sentence_bleu(
                    [ground_truth.split()], 
                    predicted.split(), 
                    smoothing_function=smoothie
                )
                metrics['bleu'] = float(bleu_score)
            except Exception as e:
                logger.warning(f"Ошибка при вычислении BLEU: {e}")
                metrics['bleu'] = 0.0
        
        # Дополнительные метрики
        metrics['length_ratio'] = len(predicted) / max(len(ground_truth), 1)
        metrics['exact_match'] = 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
        
        return metrics
    
    def _aggregate_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Агрегирует метрики по всем примерам.
        
        Args:
            predictions: Список предсказаний с метриками
            
        Returns:
            Dict[str, float]: Агрегированные метрики
        """
        if not predictions:
            return {}
        
        # Собираем все метрики
        all_metrics = {}
        for pred in predictions:
            for metric_name, value in pred['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Вычисляем статистики
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
                aggregated[f'{metric_name}_min'] = np.min(values)
                aggregated[f'{metric_name}_max'] = np.max(values)
                aggregated[f'{metric_name}_median'] = np.median(values)
        
        # Дополнительные метрики
        response_times = [pred['metrics'].get('response_time', 0) for pred in predictions]
        if response_times:
            aggregated['avg_response_time'] = np.mean(response_times)
            aggregated['total_response_time'] = np.sum(response_times)
        
        return aggregated
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Сохраняет результаты оценки в файл.
        
        Args:
            results: Результаты оценки
            output_path: Путь для сохранения
        """
        logger.info(f"Сохранение результатов оценки в: {output_path}")
        
        # Создаем директорию если нужно
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем в JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Создаем CSV с предсказаниями
        if results['predictions']:
            csv_path = output_path.replace('.json', '_predictions.csv')
            df_data = []
            for pred in results['predictions']:
                row = {
                    'question': pred['question'],
                    'ground_truth': pred['ground_truth'],
                    'predicted': pred['predicted'],
                    'sample_id': pred['sample_id']
                }
                # Добавляем метрики
                for metric_name, value in pred['metrics'].items():
                    row[metric_name] = value
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Предсказания сохранены в: {csv_path}")
    
    def compare_pipelines(self, pipeline_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Сравнивает результаты нескольких пайплайнов.
        
        Args:
            pipeline_results: Словарь с результатами пайплайнов
            
        Returns:
            Dict[str, Any]: Результаты сравнения
        """
        logger.info(f"Сравнение {len(pipeline_results)} пайплайнов")
        
        comparison = {
            'pipelines': list(pipeline_results.keys()),
            'metrics_comparison': {},
            'ranking': {}
        }
        
        # Собираем метрики всех пайплайнов
        all_metrics = set()
        for results in pipeline_results.values():
            all_metrics.update(results['metrics'].keys())
        
        # Сравниваем метрики
        for metric in all_metrics:
            if metric.endswith('_mean'):
                metric_values = {}
                for pipeline_name, results in pipeline_results.items():
                    if metric in results['metrics']:
                        metric_values[pipeline_name] = results['metrics'][metric]
                
                if metric_values:
                    comparison['metrics_comparison'][metric] = metric_values
                    
                    # Создаем рейтинг (чем больше, тем лучше)
                    sorted_pipelines = sorted(
                        metric_values.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    comparison['ranking'][metric] = [name for name, _ in sorted_pipelines]
        
        return comparison
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Генерирует текстовый отчет по результатам оценки.
        
        Args:
            results: Результаты оценки
            
        Returns:
            str: Текстовый отчет
        """
        report = []
        report.append("=" * 50)
        report.append("ОТЧЕТ ПО ОЦЕНКЕ RAG СИСТЕМЫ")
        report.append("=" * 50)
        report.append("")
        
        # Общая информация
        report.append(f"Общее количество примеров: {results['total_samples']}")
        report.append(f"Время оценки: {results['evaluation_time']:.2f} секунд")
        report.append("")
        
        # Метрики
        report.append("МЕТРИКИ КАЧЕСТВА:")
        report.append("-" * 30)
        for metric_name, value in results['metrics'].items():
            if isinstance(value, float):
                report.append(f"{metric_name}: {value:.4f}")
            else:
                report.append(f"{metric_name}: {value}")
        report.append("")
        
        # Топ-5 лучших и худших примеров по косинусному сходству
        if results['predictions']:
            predictions = results['predictions']
            cosine_scores = [
                (i, pred['metrics'].get('cosine_similarity', 0))
                for i, pred in enumerate(predictions)
            ]
            cosine_scores.sort(key=lambda x: x[1], reverse=True)
            
            report.append("ТОП-5 ЛУЧШИХ ОТВЕТОВ:")
            report.append("-" * 30)
            for i, (idx, score) in enumerate(cosine_scores[:5]):
                pred = predictions[idx]
                report.append(f"{i+1}. Сходство: {score:.4f}")
                report.append(f"   Вопрос: {pred['question'][:100]}...")
                report.append(f"   Ответ: {pred['predicted'][:100]}...")
                report.append("")
            
            report.append("ТОП-5 ХУДШИХ ОТВЕТОВ:")
            report.append("-" * 30)
            for i, (idx, score) in enumerate(cosine_scores[-5:]):
                pred = predictions[idx]
                report.append(f"{i+1}. Сходство: {score:.4f}")
                report.append(f"   Вопрос: {pred['question'][:100]}...")
                report.append(f"   Ответ: {pred['predicted'][:100]}...")
                report.append("")
        
        return "\n".join(report)


def create_evaluator(config: Dict[str, Any]) -> RAGEvaluator:
    """
    Создает и возвращает оценщик RAG системы.
    
    Args:
        config: Конфигурация из YAML файла
        
    Returns:
        RAGEvaluator: Оценщик системы
    """
    return RAGEvaluator(config)

