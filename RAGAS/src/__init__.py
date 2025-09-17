"""
RAG Система - Модульный проект для Retrieval-Augmented Generation.

Этот пакет содержит все основные компоненты RAG системы:
- models: Загрузка и управление моделями
- data_processing: Обработка данных и создание векторной базы
- pipeline: Основной RAG пайплайн
- evaluation: Система оценки качества
"""

__version__ = "1.0.0"
__author__ = "RAG Research Team"

from .models import ModelManager, create_model_manager
from .data_processing import DataProcessor, create_data_processor
from .pipeline import RAGPipeline, create_rag_pipeline, RAGPipelineManager
from .evaluation import RAGEvaluator, create_evaluator

__all__ = [
    "ModelManager",
    "create_model_manager",
    "DataProcessor", 
    "create_data_processor",
    "RAGPipeline",
    "create_rag_pipeline",
    "RAGPipelineManager",
    "RAGEvaluator",
    "create_evaluator"
]

