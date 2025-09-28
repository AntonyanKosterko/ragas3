#!/usr/bin/env python3
"""
Упрощенная система оптимизации гиперпараметров с прогресс-баром
"""

import yaml
import json
import time
import logging
import numpy as np
import os
import random
from typing import Dict, List, Any
from pathlib import Path
import mlflow
import mlflow.sklearn
from tqdm import tqdm
import optuna
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHyperparameterOptimizer:
    """Упрощенный оптимизатор гиперпараметров"""
    
    def __init__(self, base_config_path: str, experiment_name: str = "Hyperparameter_Optimization_Results"):
        """Инициализация оптимизатора"""
        self.base_config_path = base_config_path
        self.experiment_name = experiment_name
        self.base_config = self._load_config(base_config_path)
        self.best_score = 0.0
        self.best_params = None
        self.trial_results = []
        
        # Настройка MLflow
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"📊 MLflow эксперимент создан: {experiment_name}")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при создании MLflow эксперимента: {e}")
            try:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                logger.info(f"📊 MLflow эксперимент создан вручную: {experiment_name}")
            except Exception as e2:
                logger.error(f"❌ Не удалось создать MLflow эксперимент: {e2}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка базовой конфигурации"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _save_config(self, config: Dict, path: str):
        """Сохранение конфигурации"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def _create_trial_config(self, trial: optuna.Trial) -> Dict:
        """Создание конфигурации для одного испытания"""
        config = self.base_config.copy()
        
        # === ГИПЕРПАРАМЕТРЫ ДЛЯ ОПТИМИЗАЦИИ ===
        
        # 1. Веса гибридного поиска
        semantic_weight = trial.suggest_float('semantic_weight', 0.3, 0.9, step=0.1)
        bm25_weight = 1.0 - semantic_weight
        
        # 2. Количество документов
        semantic_k = trial.suggest_int('semantic_k', 8, 20, step=2)
        bm25_k = trial.suggest_int('bm25_k', 8, 20, step=2)
        final_k = trial.suggest_int('final_k', 3, 10, step=1)
        
        # 3. Параметры BM25
        bm25_k1 = trial.suggest_float('bm25_k1', 0.8, 2.0, step=0.1)
        bm25_b = trial.suggest_float('bm25_b', 0.5, 1.0, step=0.05)
        
        # 4. Размер чанков
        chunk_size = trial.suggest_categorical('chunk_size', [300, 400, 500, 600, 700, 800])
        chunk_overlap = trial.suggest_int('chunk_overlap', 30, 100, step=10)
        
        # 5. Тип поиска
        search_type = trial.suggest_categorical('search_type', ['hybrid', 'similarity', 'bm25'])
        
        # Применение параметров к конфигурации
        if 'hybrid_search' in config:
            config['hybrid_search']['semantic_weight'] = semantic_weight
            config['hybrid_search']['bm25_weight'] = bm25_weight
            config['hybrid_search']['semantic_k'] = semantic_k
            config['hybrid_search']['bm25_k'] = bm25_k
            config['hybrid_search']['final_k'] = final_k
            config['hybrid_search']['bm25_k1'] = bm25_k1
            config['hybrid_search']['bm25_b'] = bm25_b
        
        config['retriever']['search_type'] = search_type
        config['retriever']['k'] = final_k
        
        config['text_splitter']['chunk_size'] = chunk_size
        config['text_splitter']['chunk_overlap'] = chunk_overlap
        
        # Добавляем метаданные испытания
        config['trial_number'] = trial.number
        config['trial_params'] = {
            'semantic_weight': semantic_weight,
            'bm25_weight': bm25_weight,
            'semantic_k': semantic_k,
            'bm25_k': bm25_k,
            'final_k': final_k,
            'bm25_k1': bm25_k1,
            'bm25_b': bm25_b,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'search_type': search_type
        }
        
        return config
    
    def _run_single_test(self, config: Dict, max_samples: int = 50) -> Dict:
        """Запуск одного теста RAG системы"""
        try:
            # Создание временной конфигурации
            temp_config_path = f"temp_config_trial_{config['trial_number']}.yaml"
            self._save_config(config, temp_config_path)
            
            # Импорт и запуск теста напрямую
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            
            from src.pipeline import create_rag_pipeline
            from src.dataset_loader import create_dataset_loader
            from src.evaluation import RAGEvaluator
            
            # Создание компонентов
            dataset_loader = create_dataset_loader(config)
            evaluator = RAGEvaluator(config)
            
            # Получение информации о датасете
            datasets_config = config.get('datasets', {})
            dataset_name = list(datasets_config.keys())[0]
            dataset_config = datasets_config[dataset_name]
            
            # Загрузка QA пар
            qa_pairs_file = dataset_config['qa_pairs_file']
            dataset_path = dataset_config['path']
            
            # Создаем полный путь к файлу
            if os.path.isabs(qa_pairs_file):
                qa_pairs_path = qa_pairs_file
            else:
                # Если путь относительный, добавляем путь к датасету
                if os.path.isabs(dataset_path):
                    qa_pairs_path = os.path.join(dataset_path, qa_pairs_file)
                else:
                    qa_pairs_path = os.path.join(os.path.dirname(__file__), dataset_path, qa_pairs_file)
            
            with open(qa_pairs_path, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            # Ограничение количества примеров (фиксированные 100 семплов)
            max_samples = 100
            if len(qa_pairs) > max_samples:
                # Фиксируем семплы для воспроизводимости
                qa_pairs = qa_pairs[:max_samples]
            
            # Создание пайплайна
            pipeline = create_rag_pipeline(config)
            
            # Инициализация пайплайна
            pipeline.initialize()
            
            # Тестирование
            results = evaluator.evaluate_pipeline(pipeline, qa_pairs)
            
            # Очистка временного файла
            Path(temp_config_path).unlink()
            
            return {
                'success': True,
                'metrics': results['metrics'],
                'total_samples': results['total_samples']
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Целевая функция для оптимизации"""
        try:
            # Создание конфигурации для испытания
            config = self._create_trial_config(trial)
            
            # Запуск теста
            test_result = self._run_single_test(config, max_samples=50)
            
            if not test_result['success']:
                logger.error(f"❌ Тест не удался для испытания {trial.number}: {test_result.get('error', 'Unknown error')}")
                return 0.0
            
            metrics = test_result['metrics']
            
            # Вычисление целевой метрики
            mrr = metrics.get('retriever_mrr_mean', 0.0)
            precision = metrics.get('retriever_precision_mean', 0.0)
            recall = metrics.get('retriever_recall_mean', 0.0)
            ndcg = metrics.get('retriever_ndcg_mean', 0.0)
            hit_rate = metrics.get('retriever_hit_rate_mean', 0.0)
            retrieval_time = metrics.get('retrieval_time_mean', 1.0)
            
            # Комбинированная оценка
            quality_score = 0.4 * mrr + 0.2 * precision + 0.2 * recall + 0.1 * ndcg + 0.1 * hit_rate
            time_penalty = min(0.1, retrieval_time * 0.01)
            final_score = quality_score - time_penalty
            
            # Логирование в MLflow
            with mlflow.start_run(run_name=f"trial_{trial.number}"):
                # Логирование параметров
                for param_name, param_value in config['trial_params'].items():
                    mlflow.log_param(param_name, param_value)
                
                # Логирование метрик
                mlflow.log_metric("objective_score", final_score)
                mlflow.log_metric("mrr", mrr)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("ndcg", ndcg)
                mlflow.log_metric("hit_rate", hit_rate)
                mlflow.log_metric("retrieval_time", retrieval_time)
                mlflow.log_metric("quality_score", quality_score)
                
                # Логирование конфигурации
                mlflow.log_text(yaml.dump(config, default_flow_style=False), "config.yaml")
            
            # Сохранение результатов испытания
            trial_result = {
                'trial_number': trial.number,
                'params': config['trial_params'],
                'metrics': metrics,
                'objective_score': final_score,
                'quality_score': quality_score,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_results.append(trial_result)
            
            # Обновление лучшего результата
            if final_score > self.best_score:
                self.best_score = final_score
                self.best_params = config['trial_params'].copy()
                logger.info(f"🏆 Новый лучший результат! Оценка: {final_score:.4f}")
                logger.info(f"📊 MRR: {mrr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            logger.info(f"✅ Испытание {trial.number} завершено. Оценка: {final_score:.4f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Ошибка в испытании {trial.number}: {e}")
            return 0.0
    
    def optimize(self, n_trials: int = 10, timeout: int = 1800):
        """Запуск оптимизации гиперпараметров"""
        # Устанавливаем seed для воспроизводимости
        random.seed(42)
        np.random.seed(42)
        
        logger.info("🚀 Начало оптимизации гиперпараметров ретривера")
        logger.info(f"📊 Количество испытаний: {n_trials}")
        logger.info(f"⏰ Таймаут: {timeout} секунд")
        logger.info(f"🎲 Seed установлен: 42 (фиксированные семплы)")
        
        # Создание исследования Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=2,
                interval_steps=1
            )
        )
        
        # Запуск оптимизации с прогресс-баром
        logger.info("🔍 Запуск байесовского поиска...")
        
        # Создание прогресс-бара для испытаний
        pbar = tqdm(total=n_trials, desc="Оптимизация", unit="испытание")
        
        def objective_with_progress(trial):
            result = self._objective(trial)
            pbar.update(1)
            pbar.set_postfix({
                'best_score': f"{self.best_score:.4f}",
                'current_score': f"{result:.4f}"
            })
            return result
        
        try:
            study.optimize(
                objective_with_progress,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False  # Используем свой прогресс-бар
            )
        except KeyboardInterrupt:
            logger.info("⏹️ Оптимизация прервана пользователем")
        except Exception as e:
            logger.error(f"❌ Ошибка при оптимизации: {e}")
        finally:
            pbar.close()
        
        # Анализ результатов
        self._analyze_optimization_results(study)
        
        # Сохранение результатов
        self._save_optimization_results(study)
        
        logger.info("✅ Оптимизация гиперпараметров завершена")
    
    def _analyze_optimization_results(self, study: optuna.Study):
        """Анализ результатов оптимизации"""
        logger.info("\n" + "="*80)
        logger.info("📊 АНАЛИЗ РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
        logger.info("="*80)
        
        if not study.trials:
            logger.warning("⚠️ Нет завершенных испытаний для анализа")
            return
        
        # Статистика по испытаниям
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        logger.info(f"📈 Завершено испытаний: {len(completed_trials)}")
        logger.info(f"📈 Прервано испытаний: {len(study.trials) - len(completed_trials)}")
        
        if completed_trials:
            # Лучший результат
            best_trial = study.best_trial
            logger.info(f"\n🏆 ЛУЧШИЙ РЕЗУЛЬТАТ:")
            logger.info(f"   Оценка: {best_trial.value:.4f}")
            logger.info(f"   Испытание: {best_trial.number}")
            logger.info(f"   Параметры:")
            for param, value in best_trial.params.items():
                logger.info(f"     {param}: {value}")
            
            # Статистика по оценкам
            scores = [t.value for t in completed_trials if t.value is not None]
            if scores:
                logger.info(f"\n📊 СТАТИСТИКА ПО ОЦЕНКАМ:")
                logger.info(f"   Средняя оценка: {np.mean(scores):.4f}")
                logger.info(f"   Медианная оценка: {np.median(scores):.4f}")
                logger.info(f"   Стандартное отклонение: {np.std(scores):.4f}")
                logger.info(f"   Минимальная оценка: {np.min(scores):.4f}")
                logger.info(f"   Максимальная оценка: {np.max(scores):.4f}")
    
    def _save_optimization_results(self, study: optuna.Study):
        """Сохранение результатов оптимизации"""
        results_path = "results/simple_hyperparameter_optimization_results.json"
        Path("results").mkdir(exist_ok=True)
        
        # Подготовка данных для сохранения
        results_data = {
            'optimization_info': {
                'experiment_name': self.experiment_name,
                'base_config_path': self.base_config_path,
                'total_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'best_score': self.best_score,
                'best_params': self.best_params,
                'optimization_time': datetime.now().isoformat()
            },
            'study_results': {
                'best_trial': {
                    'number': study.best_trial.number,
                    'value': study.best_trial.value,
                    'params': study.best_trial.params
                } if study.best_trial else None,
                'all_trials': [
                    {
                        'number': trial.number,
                        'value': trial.value,
                        'params': trial.params,
                        'state': trial.state.name
                    }
                    for trial in study.trials
                ]
            },
            'trial_results': self.trial_results
        }
        
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
            else:
                return obj
        
        # Конвертация numpy типов
        results_data = convert_numpy_types(results_data)
        
        # Сохранение в JSON
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Результаты сохранены в {results_path}")
        
        # Создание лучшей конфигурации
        if self.best_params:
            best_config = self.base_config.copy()
            best_config.update({
                'hybrid_search': {
                    **best_config.get('hybrid_search', {}),
                    'semantic_weight': self.best_params['semantic_weight'],
                    'bm25_weight': self.best_params['bm25_weight'],
                    'semantic_k': self.best_params['semantic_k'],
                    'bm25_k': self.best_params['bm25_k'],
                    'final_k': self.best_params['final_k'],
                    'bm25_k1': self.best_params['bm25_k1'],
                    'bm25_b': self.best_params['bm25_b']
                },
                'retriever': {
                    **best_config.get('retriever', {}),
                    'search_type': self.best_params['search_type'],
                    'k': self.best_params['final_k']
                },
                'text_splitter': {
                    **best_config.get('text_splitter', {}),
                    'chunk_size': self.best_params['chunk_size'],
                    'chunk_overlap': self.best_params['chunk_overlap']
                }
            })
            
            best_config_path = "config/best_simple_retriever_config.yaml"
            self._save_config(best_config, best_config_path)
            logger.info(f"🏆 Лучшая конфигурация сохранена в {best_config_path}")

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Упрощенная оптимизация гиперпараметров ретривера")
    parser.add_argument("--config", default="config/hybrid_cpu_config.yaml", 
                       help="Путь к базовой конфигурации")
    parser.add_argument("--n-trials", type=int, default=10,
                       help="Количество испытаний для оптимизации")
    parser.add_argument("--timeout", type=int, default=1800,
                       help="Таймаут оптимизации в секундах")
    parser.add_argument("--experiment-name", default="Simple_Retriever_Optimization",
                       help="Название эксперимента в MLflow")
    
    args = parser.parse_args()
    
    # Создание оптимизатора
    optimizer = SimpleHyperparameterOptimizer(args.config, args.experiment_name)
    
    # Запуск оптимизации
    optimizer.optimize(n_trials=args.n_trials, timeout=args.timeout)

if __name__ == "__main__":
    main()
