"""
Модуль для загрузки и обработки данных в RAG системе.
Включает загрузку документов, разбиение на чанки и создание векторной базы.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    DirectoryLoader,
    UnstructuredFileLoader
)
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
# MMRRetriever может быть недоступен в некоторых версиях
# from langchain_community.retrievers.mmr import MMRRetriever
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class DataProcessor:
    """Класс для обработки данных в RAG системе."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация процессора данных.
        
        Args:
            config: Конфигурация из YAML файла
        """
        self.config = config
        self.text_splitter = None
        self.vector_store = None
        self.retriever = None
        
    def _create_text_splitter(self) -> Union[RecursiveCharacterTextSplitter, CharacterTextSplitter]:
        """
        Создает сплиттер для разбиения текста на чанки.
        
        Returns:
            Text splitter для разбиения текста
        """
        splitter_type = self.config['data'].get('text_splitter', 'recursive')
        chunk_size = self.config['data']['chunk_size']
        chunk_overlap = self.config['data']['chunk_overlap']
        
        if splitter_type == 'recursive':
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        else:
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        
        self.text_splitter = splitter
        logger.info(f"Создан {splitter_type} сплиттер: chunk_size={chunk_size}, overlap={chunk_overlap}")
        return splitter
    
    def load_documents(self, input_path: str) -> List[Document]:
        """
        Загружает документы из указанной директории или JSON файла.
        
        Args:
            input_path: Путь к директории с документами или JSON файлу
            
        Returns:
            List[Document]: Список загруженных документов
        """
        logger.info(f"Загрузка документов из: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Путь {input_path} не найден")
        
        documents = []
        
        # Проверяем, является ли путь JSON файлом
        if input_path.endswith('.json'):
            documents = self._load_documents_from_json(input_path)
        else:
            # Загружаем из директории
            documents = self._load_documents_from_directory(input_path)
        
        logger.info(f"Всего загружено документов: {len(documents)}")
        return documents
    
    def _load_documents_from_json(self, json_path: str) -> List[Document]:
        """Загружает документы из JSON файла."""
        logger.info(f"Загрузка документов из JSON: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        # Обрабатываем разные форматы JSON
        if isinstance(data, dict):
            # Формат: {id: {content: ..., metadata: ...}}
            for doc_id, doc_data in data.items():
                content = doc_data.get('content', '')
                metadata = doc_data.get('metadata', {})
                metadata['source'] = json_path
                metadata['doc_id'] = doc_id
                
                if content:
                    documents.append(Document(page_content=content, metadata=metadata))
        elif isinstance(data, list):
            # Формат: [{content: ..., metadata: ...}, ...]
            for i, doc_data in enumerate(data):
                content = doc_data.get('content', '')
                metadata = doc_data.get('metadata', {})
                metadata['source'] = json_path
                metadata['doc_id'] = i
                
                if content:
                    documents.append(Document(page_content=content, metadata=metadata))
        
        logger.info(f"Загружено {len(documents)} документов из JSON")
        return documents
    
    def _load_documents_from_directory(self, input_path: str) -> List[Document]:
        """Загружает документы из директории."""
        documents = []
        
        # Определяем типы файлов для загрузки
        file_types = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.md': TextLoader
        }
        
        # Загружаем файлы по типам
        for file_type, loader_class in file_types.items():
            try:
                loader = DirectoryLoader(
                    input_path,
                    glob=f"**/*{file_type}",
                    loader_cls=loader_class,
                    show_progress=True
                )
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Загружено {len(docs)} файлов типа {file_type}")
            except Exception as e:
                logger.warning(f"Ошибка при загрузке файлов {file_type}: {e}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Разбивает документы на чанки.
        
        Args:
            documents: Список документов для разбиения
            
        Returns:
            List[Document]: Список чанков
        """
        if not self.text_splitter:
            self._create_text_splitter()
        
        logger.info("Разбиение документов на чанки...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Создано чанков: {len(chunks)}")
        
        # Логируем статистику по размерам чанков
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        logger.info(f"Размер чанков - мин: {min(chunk_sizes)}, макс: {max(chunk_sizes)}, средний: {sum(chunk_sizes)/len(chunk_sizes):.1f}")
        
        return chunks
    
    def create_vector_store(self, chunks: List[Document], embedding_model: HuggingFaceEmbeddings) -> Union[Chroma, FAISS]:
        """
        Создает векторную базу данных из чанков.
        
        Args:
            chunks: Список чанков
            embedding_model: Модель для создания эмбеддингов
            
        Returns:
            Vector store для поиска
        """
        vector_store_config = self.config['vector_store']
        store_type = vector_store_config['type']
        persist_directory = vector_store_config['persist_directory']
        collection_name = vector_store_config.get('collection_name', 'documents')
        
        logger.info(f"Создание векторной базы типа: {store_type}")
        
        # Создаем директорию для сохранения
        os.makedirs(persist_directory, exist_ok=True)
        
        try:
            if store_type == 'chroma':
                # Настройки для ChromaDB
                chroma_settings = Settings(
                    persist_directory=persist_directory,
                    anonymized_telemetry=False
                )
                
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                    client_settings=chroma_settings
                )
                
            elif store_type == 'faiss':
                vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=embedding_model
                )
                # Сохраняем FAISS индекс
                vector_store.save_local(persist_directory)
                
            else:
                raise ValueError(f"Неподдерживаемый тип векторной базы: {store_type}")
            
            self.vector_store = vector_store
            logger.info(f"Векторная база создана и сохранена в: {persist_directory}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Ошибка при создании векторной базы: {e}")
            raise
    
    def load_vector_store(self, embedding_model: HuggingFaceEmbeddings) -> Union[Chroma, FAISS]:
        """
        Загружает существующую векторную базу данных.
        
        Args:
            embedding_model: Модель для создания эмбеддингов
            
        Returns:
            Vector store для поиска
        """
        vector_store_config = self.config['vector_store']
        store_type = vector_store_config['type']
        persist_directory = vector_store_config['persist_directory']
        collection_name = vector_store_config.get('collection_name', 'documents')
        
        logger.info(f"Загрузка векторной базы типа: {store_type}")
        
        try:
            if store_type == 'chroma':
                vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_model,
                    collection_name=collection_name
                )
                
            elif store_type == 'faiss':
                vector_store = FAISS.load_local(
                    persist_directory,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                
            else:
                raise ValueError(f"Неподдерживаемый тип векторной базы: {store_type}")
            
            self.vector_store = vector_store
            logger.info("Векторная база успешно загружена")
            return vector_store
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке векторной базы: {e}")
            raise
    
    def create_retriever(self, search_type: str = "similarity") -> BaseRetriever:
        """
        Создает ретривер для поиска релевантных документов.
        
        Args:
            search_type: Тип поиска (similarity, mmr)
            
        Returns:
            Retriever для поиска документов
        """
        if not self.vector_store:
            raise ValueError("Векторная база не инициализирована")
        
        retriever_config = self.config['retriever']
        k = retriever_config['k']
        
        logger.info(f"Создание ретривера: {search_type}, k={k}")
        
        if search_type == "similarity":
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        elif search_type == "mmr":
            # MMR поиск временно отключен из-за проблем с импортом
            logger.warning("MMR поиск временно недоступен, используется similarity поиск")
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        else:
            raise ValueError(f"Неподдерживаемый тип поиска: {search_type}")
        
        self.retriever = retriever
        logger.info("Ретривер успешно создан")
        return retriever
    
    def load_qa_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Загружает датасет вопросов и ответов для оценки.
        
        Args:
            dataset_path: Путь к файлу с датасетом
            
        Returns:
            List[Dict[str, Any]]: Список вопросов и ответов
        """
        logger.info(f"Загрузка датасета из: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Файл датасета {dataset_path} не найден")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                if dataset_path.endswith('.json'):
                    data = json.load(f)
                else:
                    raise ValueError("Поддерживаются только JSON файлы")
            
            # Проверяем формат данных
            if isinstance(data, list):
                qa_pairs = data
            elif isinstance(data, dict) and 'data' in data:
                qa_pairs = data['data']
            else:
                raise ValueError("Неверный формат датасета")
            
            logger.info(f"Загружено {len(qa_pairs)} пар вопрос-ответ")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета: {e}")
            raise
    
    def create_sample_dataset(self, output_path: str, num_samples: int = 100) -> None:
        """
        Создает пример датасета для тестирования.
        
        Args:
            output_path: Путь для сохранения датасета
            num_samples: Количество примеров
        """
        logger.info(f"Создание примера датасета с {num_samples} образцами")
        
        # Примеры вопросов и ответов на русском языке
        sample_qa = [
            {
                "question": "Что такое машинное обучение?",
                "answer": "Машинное обучение — это раздел искусственного интеллекта, который изучает алгоритмы и статистические модели, используемые компьютерными системами для выполнения задач без явных инструкций."
            },
            {
                "question": "Какие типы машинного обучения существуют?",
                "answer": "Существуют три основных типа машинного обучения: обучение с учителем (supervised learning), обучение без учителя (unsupervised learning) и обучение с подкреплением (reinforcement learning)."
            },
            {
                "question": "Что такое нейронные сети?",
                "answer": "Нейронные сети — это вычислительные системы, вдохновленные биологическими нейронными сетями, которые состоят из взаимосвязанных узлов (нейронов) и могут обучаться выполнять задачи."
            },
            {
                "question": "Что такое глубокое обучение?",
                "answer": "Глубокое обучение — это подраздел машинного обучения, основанный на искусственных нейронных сетях с множественными слоями (глубокими нейронными сетями)."
            },
            {
                "question": "Какие алгоритмы используются в машинном обучении?",
                "answer": "В машинном обучении используются различные алгоритмы: линейная регрессия, логистическая регрессия, деревья решений, случайный лес, метод опорных векторов, k-ближайших соседей, кластеризация k-means и многие другие."
            }
        ]
        
        # Расширяем датасет до нужного размера
        extended_qa = []
        for i in range(num_samples):
            extended_qa.append(sample_qa[i % len(sample_qa)])
        
        # Сохраняем датасет
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extended_qa, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Пример датасета сохранен в: {output_path}")


def create_data_processor(config: Dict[str, Any]) -> DataProcessor:
    """
    Создает и возвращает процессор данных.
    
    Args:
        config: Конфигурация из YAML файла
        
    Returns:
        DataProcessor: Процессор данных
    """
    return DataProcessor(config)
