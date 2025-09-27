"""
Модуль для загрузки и управления моделями в RAG системе.
Поддерживает как GPU, так и CPU режимы с различными типами квантизации.
"""

import torch
import logging
import psutil
import GPUtil
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
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Получает информацию об использовании GPU памяти."""
        if not torch.cuda.is_available():
            return {"gpu_memory_used": 0.0, "gpu_memory_total": 0.0, "gpu_memory_percent": 0.0}
        
        try:
            # Получаем информацию о GPU через torch
            allocated = torch.cuda.memory_allocated() / 1e9  # GB
            reserved = torch.cuda.memory_reserved() / 1e9    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            
            # Получаем информацию через GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_used = gpu.memoryUsed / 1e3  # GB
                gpu_total = gpu.memoryTotal / 1e3  # GB
                gpu_percent = gpu.memoryUtil * 100  # %
            else:
                gpu_used = allocated
                gpu_total = total
                gpu_percent = (allocated / total) * 100
            
            return {
                "gpu_memory_allocated": allocated,
                "gpu_memory_reserved": reserved,
                "gpu_memory_total": gpu_total,
                "gpu_memory_used": gpu_used,
                "gpu_memory_percent": gpu_percent,
                "gpu_utilization": gpu.load * 100 if gpus else 0.0
            }
        except Exception as e:
            logger.warning(f"Ошибка при получении информации о GPU: {e}")
            return {"gpu_memory_used": 0.0, "gpu_memory_total": 0.0, "gpu_memory_percent": 0.0}
    
    def log_gpu_memory(self, stage: str = ""):
        """Логирует текущее использование GPU памяти."""
        gpu_info = self.get_gpu_memory_info()
        if gpu_info["gpu_memory_total"] > 0:
            logger.info(f"GPU Memory {stage}: "
                       f"Used: {gpu_info['gpu_memory_used']:.2f}GB/"
                       f"{gpu_info['gpu_memory_total']:.2f}GB "
                       f"({gpu_info['gpu_memory_percent']:.1f}%), "
                       f"Allocated: {gpu_info['gpu_memory_allocated']:.2f}GB, "
                       f"Reserved: {gpu_info['gpu_memory_reserved']:.2f}GB, "
                       f"Utilization: {gpu_info['gpu_utilization']:.1f}%")
        return gpu_info
    
    def load_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        Загружает модель для создания эмбеддингов.
        
        Returns:
            HuggingFaceEmbeddings: Модель для эмбеддингов
        """
        embedding_config = self.config['models']['embedding']
        model_name = embedding_config['name']
        
        logger.info(f"Загрузка модели эмбеддингов: {model_name}")
        self.log_gpu_memory("до загрузки эмбеддингов")
        
        try:
            # Очищаем GPU память перед загрузкой
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            self.log_gpu_memory("после загрузки эмбеддингов")
            logger.info("Модель эмбеддингов успешно загружена")
            return embedding_model
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели эмбеддингов: {e}")
            # Очищаем память при ошибке
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        self.log_gpu_memory("до загрузки генеративной модели")
        
        # Проверяем, нужна ли RAG генеративная модель
        if model_name == "rag":
            from .rag_generator import RAGGenerator
            generator_model = RAGGenerator()
            self.generator_model = generator_model
            logger.info("RAG генеративная модель успешно загружена")
            return generator_model
        
        try:
            # Загружаем токенизатор
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                # Если не удается загрузить с AutoTokenizer, пробуем LlamaTokenizer
                if "mistral" in model_name.lower() or "llama" in model_name.lower():
                    from transformers import LlamaTokenizer
                    tokenizer = LlamaTokenizer.from_pretrained(model_name)
                else:
                    raise e
            
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
            
            # Загружаем модель с использованием safetensors
            model_kwargs = {
                'torch_dtype': getattr(torch, generator_config.get('torch_dtype', 'float16')),
                'device_map': 'auto' if self.device == 'cuda' else None,
                'quantization_config': quantization_config,
                'trust_remote_code': True,
                'use_safetensors': True  # Используем safetensors для безопасности
            }
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            except Exception as e:
                logger.info(f"Первая попытка загрузки модели не удалась: {e}")
                # Если ошибка связана с TensorFlow весами, пробуем с from_tf=True
                if "tensorflow" in str(e).lower() or "pytorch_model.bin" in str(e):
                    logger.info("Пробуем загрузить модель с from_tf=True")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            from_tf=True,
                            **model_kwargs
                        )
                    except Exception as e2:
                        # Если не удается загрузить с AutoModelForCausalLM, пробуем LlamaForCausalLM
                        if "mistral" in model_name.lower() or "llama" in model_name.lower():
                            from transformers import LlamaForCausalLM
                            try:
                                model = LlamaForCausalLM.from_pretrained(
                                    model_name,
                                    from_tf=True,
                                    **model_kwargs
                                )
                            except Exception as e3:
                                model = LlamaForCausalLM.from_pretrained(
                                    model_name,
                                    **model_kwargs
                                )
                        else:
                            raise e2
                else:
                    # Если не удается загрузить с AutoModelForCausalLM, пробуем LlamaForCausalLM
                    if "mistral" in model_name.lower() or "llama" in model_name.lower():
                        from transformers import LlamaForCausalLM
                        model = LlamaForCausalLM.from_pretrained(
                            model_name,
                            **model_kwargs
                        )
                    else:
                        raise e
            
            # Создаем пайплайн
            pipe_kwargs = {
                "task": "text-generation",
                "model": model,
                "tokenizer": tokenizer,
                "max_new_tokens": generator_config.get('max_new_tokens', 256),
                "temperature": generator_config.get('temperature', 0.7),
                "do_sample": generator_config.get('do_sample', True),
                "pad_token_id": tokenizer.eos_token_id,
                "truncation": True
            }
            
            # Добавляем device только если не используется accelerate
            if not hasattr(model, 'hf_device_map') or model.hf_device_map is None:
                pipe_kwargs['device'] = self.device
            
            pipe = pipeline(**pipe_kwargs)
            
            # Обертываем в LangChain
            generator_model = HuggingFacePipeline(pipeline=pipe)
            
            self.generator_model = generator_model
            self.log_gpu_memory("после загрузки генеративной модели")
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
