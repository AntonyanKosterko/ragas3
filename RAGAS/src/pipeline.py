"""
Основной RAG пайплайн для генерации ответов на основе извлеченных документов.
Включает все компоненты: загрузку данных, создание эмбеддингов, поиск и генерацию.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_community.vectorstores import Chroma, FAISS

from .models import ModelManager
from .data_processing import DataProcessor

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Основной класс RAG пайплайна."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация RAG пайплайна.
        
        Args:
            config: Конфигурация из YAML файла
        """
        self.config = config
        self.model_manager = ModelManager(config)
        self.data_processor = DataProcessor(config)
        
        # Компоненты пайплайна
        self.embedding_model = None
        self.generator_model = None
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        # Статистика
        self.stats = {
            'total_queries': 0,
            'total_time': 0.0,
            'avg_response_time': 0.0
        }
    
    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Инициализирует все компоненты пайплайна.
        
        Args:
            force_rebuild: Принудительно пересоздать векторную базу
        """
        logger.info("Инициализация RAG пайплайна...")
        
        # Загружаем модели
        self.embedding_model, self.generator_model = self.model_manager.load_models()
        
        # Проверяем существование векторной базы
        vector_store_path = self.config['vector_store']['persist_directory']
        vector_store_exists = self._check_vector_store_exists(vector_store_path)
        
        if force_rebuild or not vector_store_exists:
            logger.info("Создание новой векторной базы...")
            self._build_vector_store()
        else:
            logger.info("Загрузка существующей векторной базы...")
            self._load_vector_store()
        
        # Создаем ретривер
        retriever_config = self.config['retriever']
        search_type = retriever_config.get('search_type', 'similarity')
        self.retriever = self.data_processor.create_retriever(search_type)
        
        # Создаем QA цепочку
        self._create_qa_chain()
        
        logger.info("RAG пайплайн успешно инициализирован")
    
    def _check_vector_store_exists(self, path: str) -> bool:
        """Проверяет существование векторной базы."""
        import os
        return os.path.exists(path) and len(os.listdir(path)) > 0
    
    def _build_vector_store(self) -> None:
        """Создает векторную базу из документов."""
        # Загружаем документы
        input_path = self.config['data']['input_path']
        documents = self.data_processor.load_documents(input_path)
        
        # Разбиваем на чанки
        chunks = self.data_processor.split_documents(documents)
        
        # Создаем векторную базу
        self.vector_store = self.data_processor.create_vector_store(chunks, self.embedding_model)
    
    def _load_vector_store(self) -> None:
        """Загружает существующую векторную базу."""
        self.vector_store = self.data_processor.load_vector_store(self.embedding_model)
    
    def _create_qa_chain(self) -> None:
        """Создает цепочку для вопросов и ответов."""
        # Создаем промпт-шаблон
        prompt_template = """Используя предоставленный контекст, дайте краткий и точный ответ на вопрос.

Контекст: {context}

Вопрос: {question}

Ответ (кратко и по существу):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Создаем цепочку
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.generator_model,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("QA цепочка создана")
    
    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Выполняет запрос к RAG системе.
        
        Args:
            question: Вопрос пользователя
            return_sources: Возвращать ли исходные документы
            
        Returns:
            Dict[str, Any]: Ответ и метаданные
        """
        if not self.qa_chain:
            raise ValueError("RAG пайплайн не инициализирован")
        
        start_time = time.time()
        
        try:
            # Выполняем запрос
            result = self.qa_chain.invoke({"query": question})
            
            # Извлекаем ответ
            answer = result.get("result", "")
            
            # Извлекаем исходные документы
            source_documents = []
            if return_sources and "source_documents" in result:
                source_documents = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', None)
                    }
                    for doc in result["source_documents"]
                ]
            
            # Вычисляем время ответа
            response_time = time.time() - start_time
            
            # Обновляем статистику
            self._update_stats(response_time)
            
            response = {
                "question": question,
                "answer": answer,
                "source_documents": source_documents,
                "response_time": response_time,
                "timestamp": time.time()
            }
            
            logger.info(f"Запрос обработан за {response_time:.2f} секунд")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            return {
                "question": question,
                "answer": f"Ошибка при обработке запроса: {str(e)}",
                "source_documents": [],
                "response_time": time.time() - start_time,
                "timestamp": time.time(),
                "error": str(e)
            }
    
    def _update_stats(self, response_time: float) -> None:
        """Обновляет статистику пайплайна."""
        self.stats['total_queries'] += 1
        self.stats['total_time'] += response_time
        self.stats['avg_response_time'] = self.stats['total_time'] / self.stats['total_queries']
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику пайплайна."""
        return self.stats.copy()
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Возвращает информацию о векторной базе."""
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        info = {
            "type": self.config['vector_store']['type'],
            "persist_directory": self.config['vector_store']['persist_directory'],
            "collection_name": self.config['vector_store'].get('collection_name', 'documents')
        }
        
        # Получаем количество документов
        try:
            if hasattr(self.vector_store, '_collection'):
                info["document_count"] = self.vector_store._collection.count()
            elif hasattr(self.vector_store, 'index'):
                info["document_count"] = self.vector_store.index.ntotal
            else:
                info["document_count"] = "unknown"
        except Exception as e:
            info["document_count"] = f"error: {str(e)}"
        
        return info
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Выполняет поиск похожих документов.
        
        Args:
            query: Поисковый запрос
            k: Количество документов для возврата
            
        Returns:
            List[Dict[str, Any]]: Список похожих документов
        """
        if not self.vector_store:
            raise ValueError("Векторная база не инициализирована")
        
        try:
            # Выполняем поиск
            docs = self.vector_store.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске похожих документов: {e}")
            return []
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Обрабатывает несколько вопросов одновременно.
        
        Args:
            questions: Список вопросов
            
        Returns:
            List[Dict[str, Any]]: Список ответов
        """
        logger.info(f"Обработка {len(questions)} вопросов...")
        
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Обработка вопроса {i+1}/{len(questions)}")
            result = self.query(question)
            results.append(result)
        
        logger.info("Все вопросы обработаны")
        return results
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику пайплайна."""
        self.stats = {
            'total_queries': 0,
            'total_time': 0.0,
            'avg_response_time': 0.0
        }
        logger.info("Статистика сброшена")


def create_rag_pipeline(config: Dict[str, Any]) -> RAGPipeline:
    """
    Создает и возвращает RAG пайплайн.
    
    Args:
        config: Конфигурация из YAML файла
        
    Returns:
        RAGPipeline: RAG пайплайн
    """
    return RAGPipeline(config)


class RAGPipelineManager:
    """Менеджер для управления несколькими RAG пайплайнами."""
    
    def __init__(self):
        """Инициализация менеджера пайплайнов."""
        self.pipelines = {}
        self.active_pipeline = None
    
    def add_pipeline(self, name: str, config: Dict[str, Any]) -> RAGPipeline:
        """
        Добавляет новый пайплайн.
        
        Args:
            name: Имя пайплайна
            config: Конфигурация пайплайна
            
        Returns:
            RAGPipeline: Созданный пайплайн
        """
        pipeline = RAGPipeline(config)
        self.pipelines[name] = pipeline
        logger.info(f"Пайплайн '{name}' добавлен")
        return pipeline
    
    def get_pipeline(self, name: str) -> Optional[RAGPipeline]:
        """
        Возвращает пайплайн по имени.
        
        Args:
            name: Имя пайплайна
            
        Returns:
            RAGPipeline или None
        """
        return self.pipelines.get(name)
    
    def set_active_pipeline(self, name: str) -> bool:
        """
        Устанавливает активный пайплайн.
        
        Args:
            name: Имя пайплайна
            
        Returns:
            bool: Успешность операции
        """
        if name in self.pipelines:
            self.active_pipeline = name
            logger.info(f"Активный пайплайн установлен: {name}")
            return True
        else:
            logger.error(f"Пайплайн '{name}' не найден")
            return False
    
    def get_active_pipeline(self) -> Optional[RAGPipeline]:
        """
        Возвращает активный пайплайн.
        
        Returns:
            RAGPipeline или None
        """
        if self.active_pipeline:
            return self.pipelines.get(self.active_pipeline)
        return None
    
    def list_pipelines(self) -> List[str]:
        """
        Возвращает список имен всех пайплайнов.
        
        Returns:
            List[str]: Список имен пайплайнов
        """
        return list(self.pipelines.keys())
    
    def remove_pipeline(self, name: str) -> bool:
        """
        Удаляет пайплайн.
        
        Args:
            name: Имя пайплайна
            
        Returns:
            bool: Успешность операции
        """
        if name in self.pipelines:
            del self.pipelines[name]
            if self.active_pipeline == name:
                self.active_pipeline = None
            logger.info(f"Пайплайн '{name}' удален")
            return True
        else:
            logger.error(f"Пайплайн '{name}' не найден")
            return False
