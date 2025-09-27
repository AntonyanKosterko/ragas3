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
        # Создаем модернизированный промпт-шаблон
        prompt_template = """Ты - эксперт по анализу текстов. Твоя задача: найти точный ответ на вопрос, используя предоставленный контекст.

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ИНСТРУКЦИИ:
1. Проанализируй контекст и найди информацию, которая отвечает на вопрос
2. Дай ТОЧНЫЙ ответ одним словом или короткой фразой
3. Используй ТОЛЬКО информацию из контекста
4. НЕ добавляй объяснения, примеры или дополнительную информацию
5. Если ответ содержит несколько элементов, перечисли их через запятую

ОТВЕТ:"""
        
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
        Выполняет запрос к RAG системе с детальным профилированием.
        
        Args:
            question: Вопрос пользователя
            return_sources: Возвращать ли исходные документы
            
        Returns:
            Dict[str, Any]: Ответ и метаданные с временными метриками
        """
        if not self.qa_chain:
            raise ValueError("RAG пайплайн не инициализирован")
        
        # Инициализируем временные метрики
        timing_metrics = {}
        total_start_time = time.time()
        
        try:
            # Проверяем валидность вопроса
            if not question or len(question.strip()) == 0:
                raise ValueError("Пустой вопрос")
            
            # Этап 1: Поиск релевантных документов
            retrieval_start = time.time()
            retriever_config = self.config['retriever']
            k = retriever_config.get('k', 5)
            
            # Выполняем поиск документов с защитой от ошибок
            try:
                retrieved_docs = self.retriever.get_relevant_documents(question)
                if not retrieved_docs:
                    retrieved_docs = []
            except Exception as e:
                logger.warning(f"Ошибка при поиске документов: {e}")
                retrieved_docs = []
            
            timing_metrics['retrieval_time'] = time.time() - retrieval_start
            timing_metrics['retrieved_docs_count'] = len(retrieved_docs)
            
            # Этап 2: Подготовка контекста
            context_start = time.time()
            if retrieved_docs:
                context = "\n\n".join([doc.page_content for doc in retrieved_docs if hasattr(doc, 'page_content')])
            else:
                context = "Контекст не найден."
            timing_metrics['context_preparation_time'] = time.time() - context_start
            timing_metrics['context_length'] = len(context)
            
            # Этап 3: Генерация ответа
            generation_start = time.time()
            try:
                result = self.qa_chain.invoke({"query": question})
            except Exception as e:
                logger.warning(f"Ошибка при генерации ответа: {e}")
                result = {"result": f"Ошибка генерации: {str(e)}"}
            timing_metrics['generation_time'] = time.time() - generation_start
            
            # Извлекаем ответ и очищаем его от промпта
            full_result = result.get("result", "")
            
            # Извлекаем только ответ после "Ответь кратко одним словом или короткой фразой:"
            answer_prompt = "Ответь кратко одним словом или короткой фразой:"
            if answer_prompt in full_result:
                answer = full_result.split(answer_prompt)[-1].strip()
            else:
                # Если промпт не найден, берем последнюю строку
                lines = full_result.strip().split('\n')
                answer = lines[-1].strip() if lines else full_result
            
            # Дополнительная очистка ответа для улучшения BLEU
            answer = self._clean_answer(answer)
            
            timing_metrics['answer_length'] = len(answer)
            
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
            
            # Общее время
            total_time = time.time() - total_start_time
            timing_metrics['total_time'] = total_time
            
            # Обновляем статистику
            self._update_stats(total_time, timing_metrics)
            
            response = {
                "question": question,
                "answer": answer,
                "source_documents": source_documents,
                "response_time": total_time,
                "timing_metrics": timing_metrics,
                "timestamp": time.time()
            }
            
            logger.info(f"Запрос обработан за {total_time:.2f} сек (ретривер: {timing_metrics['retrieval_time']:.2f}с, генерация: {timing_metrics['generation_time']:.2f}с)")
            return response
            
        except Exception as e:
            total_time = time.time() - total_start_time
            logger.error(f"Ошибка при обработке запроса: {e}")
            return {
                "question": question,
                "answer": f"Ошибка при обработке запроса: {str(e)}",
                "source_documents": [],
                "response_time": total_time,
                "timing_metrics": timing_metrics,
                "timestamp": time.time(),
                "error": str(e)
            }
    
    def _update_stats(self, response_time: float, timing_metrics: Dict[str, Any] = None) -> None:
        """Обновляет статистику пайплайна с детальными метриками."""
        self.stats['total_queries'] += 1
        self.stats['total_time'] += response_time
        self.stats['avg_response_time'] = self.stats['total_time'] / self.stats['total_queries']
        
        # Добавляем детальные временные метрики
        if timing_metrics:
            if 'detailed_timing' not in self.stats:
                self.stats['detailed_timing'] = {
                    'retrieval_times': [],
                    'generation_times': [],
                    'context_prep_times': [],
                    'retrieved_docs_counts': [],
                    'context_lengths': [],
                    'answer_lengths': []
                }
            
            # Собираем метрики для агрегации
            self.stats['detailed_timing']['retrieval_times'].append(timing_metrics.get('retrieval_time', 0))
            self.stats['detailed_timing']['generation_times'].append(timing_metrics.get('generation_time', 0))
            self.stats['detailed_timing']['context_prep_times'].append(timing_metrics.get('context_preparation_time', 0))
            self.stats['detailed_timing']['retrieved_docs_counts'].append(timing_metrics.get('retrieved_docs_count', 0))
            self.stats['detailed_timing']['context_lengths'].append(timing_metrics.get('context_length', 0))
            self.stats['detailed_timing']['answer_lengths'].append(timing_metrics.get('answer_length', 0))
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику пайплайна с агрегированными метриками."""
        import numpy as np
        
        stats = self.stats.copy()
        
        # Вычисляем агрегированные метрики для детального профилирования
        if 'detailed_timing' in stats and stats['detailed_timing']['retrieval_times']:
            detailed = stats['detailed_timing']
            
            # Средние времена
            stats['avg_retrieval_time'] = np.mean(detailed['retrieval_times'])
            stats['avg_generation_time'] = np.mean(detailed['generation_times'])
            stats['avg_context_prep_time'] = np.mean(detailed['context_prep_times'])
            
            # Медианные времена
            stats['median_retrieval_time'] = np.median(detailed['retrieval_times'])
            stats['median_generation_time'] = np.median(detailed['generation_times'])
            stats['median_context_prep_time'] = np.median(detailed['context_prep_times'])
            
            # Максимальные времена
            stats['max_retrieval_time'] = np.max(detailed['retrieval_times'])
            stats['max_generation_time'] = np.max(detailed['generation_times'])
            stats['max_context_prep_time'] = np.max(detailed['context_prep_times'])
            
            # Стандартные отклонения
            stats['std_retrieval_time'] = np.std(detailed['retrieval_times'])
            stats['std_generation_time'] = np.std(detailed['generation_times'])
            stats['std_context_prep_time'] = np.std(detailed['context_prep_times'])
            
            # Средние количества и длины
            stats['avg_retrieved_docs'] = np.mean(detailed['retrieved_docs_counts'])
            stats['avg_context_length'] = np.mean(detailed['context_lengths'])
            stats['avg_answer_length'] = np.mean(detailed['answer_lengths'])
            
            # Процентное распределение времени
            total_avg_time = stats['avg_retrieval_time'] + stats['avg_generation_time'] + stats['avg_context_prep_time']
            if total_avg_time > 0:
                stats['retrieval_time_percentage'] = (stats['avg_retrieval_time'] / total_avg_time) * 100
                stats['generation_time_percentage'] = (stats['avg_generation_time'] / total_avg_time) * 100
                stats['context_prep_time_percentage'] = (stats['avg_context_prep_time'] / total_avg_time) * 100
        
        return stats
    
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
    
    def _clean_answer(self, answer: str) -> str:
        """
        Очищает ответ от лишних символов и текста для улучшения метрик.
        
        Args:
            answer: Исходный ответ
            
        Returns:
            str: Очищенный ответ
        """
        if not answer:
            return ""
        
        # Убираем кавычки в начале и конце
        answer = answer.strip('"\'')
        
        # Убираем лишние символы форматирования
        answer = answer.replace('---', '').replace('### Ответ', '').replace('###', '')
        
        # Разбиваем на строки и берем только первую значимую строку
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        
        if not lines:
            return ""
        
        # Берем первую строку, которая не является служебной
        first_line = lines[0]
        
        # Убираем служебные фразы (обновленная логика для нового промпта)
        stop_phrases = [
            "ОТВЕТ:",
            "Ответ:",
            "Ответь ТОЛЬКО одним словом или короткой фразой, без дополнительных объяснений:",
            "Ответь кратко одним словом или короткой фразой:",
            "Вопрос:", "Используйте информацию", "Контекст:", 
            "Ответь кратко", "один словом", "короткой фразой",
            "Ты - эксперт по анализу текстов", "ИНСТРУКЦИИ:", "КОНТЕКСТ:", "ВОПРОС:"
        ]
        
        # Ищем и удаляем служебные фразы
        for phrase in stop_phrases:
            if phrase in first_line:
                # Берем текст после служебной фразы
                parts = first_line.split(phrase)
                if len(parts) > 1:
                    first_line = parts[1].strip()
                else:
                    # Если фраза в конце, берем текст до неё
                    first_line = parts[0].strip()
                break
        
        # Если ответ слишком короткий (меньше 2 символов), пробуем следующую строку
        if len(first_line) < 2 and len(lines) > 1:
            for line in lines[1:]:
                if len(line.strip()) >= 2:
                    first_line = line.strip()
                    break
        
        # Ограничиваем длину ответа (максимум 200 символов)
        if len(first_line) > 200:
            first_line = first_line[:200].strip()
        
        # Убираем лишние пробелы и переносы строк
        first_line = ' '.join(first_line.split())
        
        return first_line


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
