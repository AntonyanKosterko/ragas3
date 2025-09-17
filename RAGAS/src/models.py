"""
Модуль для загрузки и управления моделями в RAG системе.
Поддерживает как GPU, так и CPU режимы с различными типами квантизации.
"""

import torch
import logging
from typing import Dict, Any, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Менеджер для загрузки и управления моделями."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера моделей.
        
        Args:
            config: Конфигурация моделей из YAML файла
        """
        self.config = config
        self.device = self._get_device()
        self.embedding_model = None
        self.generator_model = None
        
    def _get_device(self) -> str:
        """Определяет доступное устройство (cuda/cpu)."""
        if torch.cuda.is_available() and self.config.get('device', 'auto') != 'cpu':
            device = 'cuda'
            logger.info(f"Используется GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Доступно VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = 'cpu'
            logger.info("Используется CPU")
        
        return device
    
    def load_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        Загружает модель для создания эмбеддингов.
        
        Returns:
            HuggingFaceEmbeddings: Модель для эмбеддингов
        """
        embedding_config = self.config['models']['embedding']
        model_name = embedding_config['name']
        
        logger.info(f"Загрузка модели эмбеддингов: {model_name}")
        
        try:
            # Создаем конфигурацию для модели
            model_kwargs = {
                'device': self.device
            }
            
            # Если используем GPU, добавляем дополнительные параметры
            if self.device == 'cuda':
                model_kwargs['model_kwargs'] = {
                    'torch_dtype': torch.float16,
                    'device_map': 'auto'
                }
            
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': embedding_config.get('normalize_embeddings', True)}
            )
            
            self.embedding_model = embedding_model
            logger.info("Модель эмбеддингов успешно загружена")
            return embedding_model
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели эмбеддингов: {e}")
            raise
    
    def load_generator_model(self) -> BaseLanguageModel:
        """
        Загружает генеративную языковую модель.
        
        Returns:
            BaseLanguageModel: Генеративная модель
        """
        generator_config = self.config['models']['generator']
        model_name = generator_config['name']
        
        logger.info(f"Загрузка генеративной модели: {model_name}")
        
        # Проверяем, нужна ли RAG генеративная модель
        if model_name == "rag":
            from .rag_generator import RAGGenerator
            generator_model = RAGGenerator()
            self.generator_model = generator_model
            logger.info("RAG генеративная модель успешно загружена")
            return generator_model
        
        try:
            # Загружаем токенизатор
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Настраиваем квантизацию для CPU
            quantization_config = None
            if self.device == 'cpu' and 'quantization_config' in generator_config:
                quant_config = generator_config['quantization_config']
                if quant_config.get('load_in_8bit', False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=quant_config.get('llm_int8_threshold', 6.0)
                    )
            
            # Загружаем модель
            model_kwargs = {
                'torch_dtype': getattr(torch, generator_config.get('torch_dtype', 'float32')),
                'device_map': 'auto' if self.device == 'cuda' else None,
                'quantization_config': quantization_config
            }
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Создаем пайплайн
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=generator_config.get('max_new_tokens', 256),
                temperature=generator_config.get('temperature', 0.7),
                do_sample=generator_config.get('do_sample', True),
                pad_token_id=tokenizer.eos_token_id,
                device=self.device,
                truncation=True
            )
            
            # Обертываем в LangChain
            generator_model = HuggingFacePipeline(pipeline=pipe)
            
            self.generator_model = generator_model
            logger.info("Генеративная модель успешно загружена")
            return generator_model
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке генеративной модели: {e}")
            raise
    
    def load_models(self) -> tuple:
        """
        Загружает все необходимые модели.
        
        Returns:
            tuple: (embedding_model, generator_model)
        """
        embedding_model = self.load_embedding_model()
        generator_model = self.load_generator_model()
        
        return embedding_model, generator_model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о загруженных моделях.
        
        Returns:
            Dict[str, Any]: Информация о моделях
        """
        info = {
            'device': self.device,
            'embedding_model': self.config['embedding']['name'] if self.embedding_model else None,
            'generator_model': self.config['generator']['name'] if self.generator_model else None,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            info['cuda_memory_allocated'] = torch.cuda.memory_allocated() / 1e9
            info['cuda_memory_reserved'] = torch.cuda.memory_reserved() / 1e9
        
        return info


def create_model_manager(config: Dict[str, Any]) -> ModelManager:
    """
    Создает и возвращает менеджер моделей.
    
    Args:
        config: Конфигурация из YAML файла
        
    Returns:
        ModelManager: Менеджер моделей
    """
    return ModelManager(config)


def load_quantized_model(model_name: str, device: str = "cpu") -> tuple:
    """
    Загружает квантизованную модель для CPU.
    
    Args:
        model_name: Название модели
        device: Устройство для загрузки
        
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Проверяем доступность bitsandbytes
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        except ImportError:
            logger.warning("bitsandbytes не установлен, используем обычную загрузку модели")
            quantization_config = None
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {
            "device_map": "auto" if device == "cuda" else None,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке квантизованной модели: {e}")
        raise


def get_available_models() -> Dict[str, list]:
    """
    Возвращает список доступных моделей для разных задач.
    
    Returns:
        Dict[str, list]: Словарь с доступными моделями
    """
    return {
        'embedding_models': [
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'sentence-transformers/distiluse-base-multilingual-cased',
            'sentence-transformers/LaBSE'
        ],
        'generator_models': [
            'microsoft/DialoGPT-small',
            'microsoft/DialoGPT-medium',
            'microsoft/DialoGPT-large',
            'gpt2',
            'gpt2-medium',
            'gpt2-large'
        ]
    }
