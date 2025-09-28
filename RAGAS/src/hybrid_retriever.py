"""
Гибридный ретривер, объединяющий семантический и ключевой поиск (BM25).
Оптимизирован для работы как на CPU, так и на GPU.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# Для BM25 поиска
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank_bm25 не установлен. Установите: pip install rank-bm25")

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Конфигурация для гибридного поиска."""
    # Семантический поиск
    semantic_weight: float = 0.7  # Вес семантического поиска (0.0-1.0)
    semantic_k: int = 10  # Количество документов от семантического поиска
    
    # BM25 поиск
    bm25_weight: float = 0.3  # Вес BM25 поиска (0.0-1.0)
    bm25_k: int = 10  # Количество документов от BM25 поиска
    
    # Общие параметры
    final_k: int = 5  # Финальное количество документов
    rerank: bool = True  # Переранжирование результатов
    normalize_scores: bool = True  # Нормализация скоров
    
    # Параметры BM25
    bm25_k1: float = 1.2  # Параметр k1 для BM25
    bm25_b: float = 0.75  # Параметр b для BM25


class HybridRetriever(BaseRetriever):
    """
    Гибридный ретривер, объединяющий семантический поиск и BM25.
    
    Подход:
    1. Семантический поиск: находит документы по смыслу
    2. BM25 поиск: находит документы по ключевым словам
    3. Объединение: комбинирует результаты с весами
    4. Переранжирование: финальная сортировка по комбинированному скору
    """
    
    def __init__(
        self,
        vector_store: Any,
        documents: List[Document],
        config: HybridSearchConfig,
        embedding_model: HuggingFaceEmbeddings
    ):
        """
        Инициализация гибридного ретривера.
        
        Args:
            vector_store: Векторная база для семантического поиска
            documents: Список документов для BM25
            config: Конфигурация гибридного поиска
            embedding_model: Модель эмбеддингов
        """
        # Инициализируем BaseRetriever с пустыми параметрами
        super().__init__()
        
        # Сохраняем параметры как атрибуты
        self._vector_store = vector_store
        self._documents = documents
        self._config = config
        self._embedding_model = embedding_model
        
        # Инициализация BM25
        self._bm25 = None
        self._initialize_bm25()
        
        # Создание индекса документов для быстрого доступа
        self._doc_id_to_doc = {doc.metadata.get('doc_id', i): doc for i, doc in enumerate(documents)}
        
        logger.info(f"Гибридный ретривер инициализирован: semantic_weight={config.semantic_weight}, "
                   f"bm25_weight={config.bm25_weight}, final_k={config.final_k}")
    
    def _initialize_bm25(self) -> None:
        """Инициализирует BM25 индекс."""
        if not BM25_AVAILABLE:
            logger.warning("BM25 недоступен, используется только семантический поиск")
            return
        
        try:
            # Подготовка текстов для BM25
            texts = []
            for doc in self._documents:
                # Простая токенизация (можно улучшить)
                text = doc.page_content.lower()
                # Убираем пунктуацию и разбиваем на слова
                import re
                tokens = re.findall(r'\b\w+\b', text)
                texts.append(tokens)
            
            # Создание BM25 индекса
            self._bm25 = BM25Okapi(texts, k1=self._config.bm25_k1, b=self._config.bm25_b)
            logger.info(f"BM25 индекс создан для {len(texts)} документов")
            
        except Exception as e:
            logger.error(f"Ошибка при создании BM25 индекса: {e}")
            self._bm25 = None
    
    def _semantic_search(self, query: str) -> List[Tuple[Document, float]]:
        """Выполняет семантический поиск."""
        try:
            # Получаем больше документов для лучшего объединения
            semantic_docs = self._vector_store.similarity_search_with_score(
                query, k=self._config.semantic_k
            )
            
            # Конвертируем в нужный формат
            results = []
            for doc, score in semantic_docs:
                # Инвертируем скор (меньше расстояние = больше релевантность)
                relevance_score = 1.0 / (1.0 + score)
                results.append((doc, relevance_score))
            
            logger.debug(f"Семантический поиск: найдено {len(results)} документов")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            return []
    
    def _bm25_search(self, query: str) -> List[Tuple[Document, float]]:
        """Выполняет BM25 поиск."""
        if not self._bm25:
            return []
        
        try:
            # Токенизация запроса
            import re
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            
            # Получение скоров BM25
            bm25_scores = self._bm25.get_scores(query_tokens)
            
            # Сортировка по убыванию скора
            doc_indices = np.argsort(bm25_scores)[::-1]
            
            results = []
            for i, doc_idx in enumerate(doc_indices[:self._config.bm25_k]):
                if bm25_scores[doc_idx] > 0:  # Только документы с положительным скором
                    doc = self._documents[doc_idx]
                    # Нормализация скора BM25
                    normalized_score = bm25_scores[doc_idx] / (bm25_scores[doc_idx] + 1.0)
                    results.append((doc, normalized_score))
            
            logger.debug(f"BM25 поиск: найдено {len(results)} документов")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка BM25 поиска: {e}")
            return []
    
    def _combine_results(
        self, 
        semantic_results: List[Tuple[Document, float]], 
        bm25_results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Объединяет результаты семантического и BM25 поиска."""
        
        # Словарь для хранения объединенных скоров
        combined_scores = {}
        
        # Добавляем семантические скоры
        for doc, score in semantic_results:
            doc_id = doc.metadata.get('doc_id', id(doc))
            combined_scores[doc_id] = {
                'doc': doc,
                'semantic_score': score,
                'bm25_score': 0.0
            }
        
        # Добавляем BM25 скоры
        for doc, score in bm25_results:
            doc_id = doc.metadata.get('doc_id', id(doc))
            if doc_id in combined_scores:
                combined_scores[doc_id]['bm25_score'] = score
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'semantic_score': 0.0,
                    'bm25_score': score
                }
        
        # Вычисляем комбинированные скоры
        final_results = []
        for doc_id, scores in combined_scores.items():
            # Взвешенная комбинация скоров
            combined_score = (
                self._config.semantic_weight * scores['semantic_score'] +
                self._config.bm25_weight * scores['bm25_score']
            )
            
            final_results.append((scores['doc'], combined_score))
        
        # Сортировка по убыванию комбинированного скора
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Объединение результатов: {len(final_results)} документов")
        return final_results
    
    def _normalize_scores(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Нормализует скоры для лучшего распределения."""
        if not results or not self._config.normalize_scores:
            return results
        
        scores = [score for _, score in results]
        if not scores:
            return results
        
        # Min-max нормализация
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return results
        
        normalized_results = []
        for doc, score in results:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_results.append((doc, normalized_score))
        
        return normalized_results
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Основной метод для получения релевантных документов.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Список релевантных документов
        """
        start_time = time.time()
        
        try:
            # 1. Семантический поиск
            semantic_results = self._semantic_search(query)
            
            # 2. BM25 поиск
            bm25_results = self._bm25_search(query)
            
            # 3. Объединение результатов
            combined_results = self._combine_results(semantic_results, bm25_results)
            
            # 4. Нормализация скоров
            if self._config.normalize_scores:
                combined_results = self._normalize_scores(combined_results)
            
            # 5. Возврат топ-k документов
            final_docs = [doc for doc, _ in combined_results[:self._config.final_k]]
            
            search_time = time.time() - start_time
            logger.debug(f"Гибридный поиск завершен за {search_time:.3f}с, найдено {len(final_docs)} документов")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"Ошибка в гибридном поиске: {e}")
            # Fallback к семантическому поиску
            try:
                fallback_docs = self._vector_store.similarity_search(query, k=self._config.final_k)
                logger.warning("Использован fallback к семантическому поиску")
                return fallback_docs
            except Exception as fallback_error:
                logger.error(f"Ошибка fallback поиска: {fallback_error}")
                return []
    
    def get_relevant_documents_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Получает релевантные документы со скорами для анализа.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Список кортежей (документ, скор)
        """
        start_time = time.time()
        
        try:
            # 1. Семантический поиск
            semantic_results = self._semantic_search(query)
            
            # 2. BM25 поиск
            bm25_results = self._bm25_search(query)
            
            # 3. Объединение результатов
            combined_results = self._combine_results(semantic_results, bm25_results)
            
            # 4. Нормализация скоров
            if self._config.normalize_scores:
                combined_results = self._normalize_scores(combined_results)
            
            # 5. Возврат топ-k документов со скорами
            final_results = combined_results[:self._config.final_k]
            
            search_time = time.time() - start_time
            logger.debug(f"Гибридный поиск со скорами завершен за {search_time:.3f}с")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка в гибридном поиске со скорами: {e}")
            return []


def create_hybrid_retriever(
    vector_store: Any,
    documents: List[Document],
    config: Dict[str, Any],
    embedding_model: HuggingFaceEmbeddings
) -> HybridRetriever:
    """
    Фабричная функция для создания гибридного ретривера.
    
    Args:
        vector_store: Векторная база
        documents: Список документов
        config: Конфигурация из YAML
        embedding_model: Модель эмбеддингов
        
    Returns:
        Настроенный гибридный ретривер
    """
    # Извлекаем конфигурацию гибридного поиска
    hybrid_config = config.get('hybrid_search', {})
    
    # Создаем конфигурацию
    search_config = HybridSearchConfig(
        semantic_weight=hybrid_config.get('semantic_weight', 0.7),
        semantic_k=hybrid_config.get('semantic_k', 10),
        bm25_weight=hybrid_config.get('bm25_weight', 0.3),
        bm25_k=hybrid_config.get('bm25_k', 10),
        final_k=hybrid_config.get('final_k', 5),
        rerank=hybrid_config.get('rerank', True),
        normalize_scores=hybrid_config.get('normalize_scores', True),
        bm25_k1=hybrid_config.get('bm25_k1', 1.2),
        bm25_b=hybrid_config.get('bm25_b', 0.75)
    )
    
    return HybridRetriever(vector_store, documents, search_config, embedding_model)
