#!/usr/bin/env python3
"""
Быстрая оптимизация параметров поиска FAISS
Оптимизирует только runtime параметры без пересборки векторной БД:
- nprobe (для IVF индексов)
- efSearch (для HNSW индексов)
"""

import os
import yaml
import mlflow
import subprocess
import tempfile
import json
from optuna import create_study, Trial
from optuna.samplers import TPESampler
import sys

def load_config(config_path: str):
    """Загрузка конфигурации"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def objective(trial: Trial, base_config_path: str) -> float:
    """Целевая функция для оптимизации параметров поиска FAISS"""
    
    # Параметры поиска для оптимизации
    nprobe = trial.suggest_int('nprobe', 1, 50, step=1)
    efSearch = trial.suggest_int('efSearch', 50, 500, step=50)
    
    # Загружаем базовую конфигурацию
    config = load_config(base_config_path)
    
    # Обновляем параметры поиска
    config['vector_store']['nprobe'] = nprobe
    config['vector_store']['efSearch'] = efSearch
    
    # Создаем временный файл конфигурации
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        temp_config_path = f.name
    
    try:
        # НЕ удаляем векторную БД - используем существующую!
        # Только обновляем параметры поиска в конфигурации
        
        # Запускаем тест БЕЗ пересборки векторной БД
        result = subprocess.run([
            'python', 'test_rag.py',
            '--config', temp_config_path,
            '--max-samples', '100'
            # НЕ добавляем --rebuild-vector-db!
        ], capture_output=True, text=True, timeout=120)  # Быстрый таймаут
        
        if result.returncode != 0:
            print(f"❌ Ошибка в trial {trial.number}: {result.stderr}")
            return 0.0
        
        # Парсим результаты из stdout
        output_lines = result.stdout.splitlines()
        metrics = {}
        for line in output_lines:
            if "MLflow Metrics:" in line:
                try:
                    metrics_str = line.split("MLflow Metrics:")[1].strip()
                    metrics = json.loads(metrics_str)
                    break
                except json.JSONDecodeError:
                    print(f"❌ Ошибка парсинга JSON метрик в trial {trial.number}")
                    continue
        
        # Если не найдено в stdout, ищем в stderr
        if not metrics:
            error_lines = result.stderr.splitlines()
            for line in error_lines:
                if "MLflow Metrics:" in line:
                    try:
                        metrics_str = line.split("MLflow Metrics:")[1].strip()
                        metrics = json.loads(metrics_str)
                        break
                    except json.JSONDecodeError:
                        print(f"❌ Ошибка парсинга JSON метрик в trial {trial.number}")
                        continue
        
        if not metrics:
            print(f"❌ Метрики не найдены в выводе test_rag.py для trial {trial.number}")
            return 0.0
        
        # Вычисляем objective score (фокус на метриках ретривера)
        objective_score = (
            metrics.get('retriever_f1_mean', 0.0) + 
            metrics.get('retriever_mrr_mean', 0.0) + 
            metrics.get('retriever_ndcg_mean', 0.0)
        ) / 3
        
        # Логируем метрики в MLflow
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params({
                'nprobe': nprobe,
                'efSearch': efSearch,
                'rebuild_vector_db': False,
                'optimization_type': 'search_parameters_only'
            })
            mlflow.log_metrics({
                'objective_score': objective_score,
                'retriever_precision': metrics.get('retriever_precision_mean', 0.0),
                'retriever_recall': metrics.get('retriever_recall_mean', 0.0),
                'retriever_f1': metrics.get('retriever_f1_mean', 0.0),
                'retriever_hit_rate': metrics.get('retriever_hit_rate_mean', 0.0),
                'retriever_mrr': metrics.get('retriever_mrr_mean', 0.0),
                'retriever_ndcg': metrics.get('retriever_ndcg_mean', 0.0),
                'retrieval_time': metrics.get('retrieval_time_mean', 0.0),
                'response_time': metrics.get('response_time_mean', 0.0),
                'generation_time': metrics.get('generation_time_mean', 0.0)
            })
        
        print(f"✅ Trial {trial.number}: nprobe={nprobe}, efSearch={efSearch}, objective_score = {objective_score:.4f}")
        return objective_score
        
    except subprocess.TimeoutExpired:
        print(f"❌ Trial {trial.number} timed out after 120 seconds.")
        return 0.0
    except Exception as e:
        print(f"❌ Ошибка в trial {trial.number}: {e}")
        return 0.0
    finally:
        os.unlink(temp_config_path)

def main():
    """Основная функция"""
    
    # Настраиваем MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("FAISS_Search_Parameters_Optimization")
    
    print("🚀 Быстрая оптимизация параметров поиска FAISS")
    print("📊 Эксперимент: FAISS_Search_Parameters_Optimization")
    print("🎯 Сэмплы: 100")
    print("🔧 Параметры: nprobe, efSearch")
    print("⚠️  Пересборка векторной БД: НЕТ")
    print("⏱️  Ожидаемое время: ~10 минут")
    print()
    
    # Создаем исследование
    study = create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    # Запускаем оптимизацию
    with mlflow.start_run(run_name="FAISS_Search_Parameters_Optimization"):
        mlflow.log_param("max_samples", 100)
        mlflow.log_param("rebuild_vector_db", False)
        mlflow.log_param("optimization_target", "faiss_search_parameters")
        mlflow.log_param("optimization_type", "search_parameters_only")
        
        study.optimize(
            lambda trial: objective(trial, "config/faiss_search_optimization.yaml"),
            n_trials=5,  # Меньше итераций для быстрого тестирования
            timeout=600  # 10 минут максимум
        )
        
        # Логируем лучшие результаты
        best_trial = study.best_trial
        mlflow.log_metric('best_objective_score', best_trial.value)
        mlflow.log_params(best_trial.params)
        
        print("\n🏆 Лучшие параметры поиска:")
        print(f"  Objective Score: {best_trial.value:.4f}")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Сохраняем лучшие параметры
        best_params = best_trial.params.copy()
        best_params['objective_score'] = best_trial.value
        
        with open('best_search_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n💾 Лучшие параметры сохранены в best_search_params.json")
        print("🎉 Оптимизация параметров поиска завершена!")
        
        # Показываем детальные результаты
        print(f"\n📊 Детальные результаты:")
        print(f"  Лучший objective score: {best_trial.value:.4f}")
        print(f"  Лучший nprobe: {best_trial.params['nprobe']}")
        print(f"  Лучший efSearch: {best_trial.params['efSearch']}")
        print(f"  Всего trials: {len(study.trials)}")

if __name__ == "__main__":
    main()
