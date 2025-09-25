"""
Модуль для загрузки документов из датасетов в векторную БД
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Класс для загрузки документов из датасетов в векторную БД"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация загрузчика датасетов.
        
        Args:
            config: Конфигурация из YAML файла
        """
        self.config = config
        self.embedding_model = None
        
    def load_embedding_model(self) -> HuggingFaceEmbeddings:
        """Загружает модель эмбеддингов"""
        if self.embedding_model is None:
            from .models import ModelManager
            model_manager = ModelManager(self.config)
            self.embedding_model = model_manager.load_embedding_model()
        return self.embedding_model
    
    def load_documents_from_json(self, json_path: str) -> List[Document]:
        """
        Загружает документы из JSON файла.
        
        Args:
            json_path: Путь к JSON файлу с документами
            
        Returns:
            List[Document]: Список документов LangChain
        """
        logger.info(f"Загрузка документов из {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Файл {json_path} не найден")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)
        
        documents = []
        
        # Обрабатываем как словарь, так и список
        if isinstance(documents_data, dict):
            for doc_id, doc_data in documents_data.items():
                content = doc_data.get('content', '')
                metadata = doc_data.get('metadata', {})
                metadata['doc_id'] = doc_id
                
                if content:
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
        elif isinstance(documents_data, list):
            for i, doc_data in enumerate(documents_data):
                content = doc_data.get('content', '')
                metadata = doc_data.get('metadata', {})
                doc_id = doc_data.get('id', f'doc_{i}')
                metadata['doc_id'] = doc_id
                
                if content:
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
        
        logger.info(f"Загружено {len(documents)} документов")
        return documents
    
    def create_vector_store_from_dataset(self, dataset_path: str, vector_store_path: str) -> None:
        """
        Создает векторную БД из датасета.
        
        Args:
            dataset_path: Путь к папке с датасетом
            vector_store_path: Путь для сохранения векторной БД
        """
        logger.info(f"Создание векторной БД из датасета {dataset_path}")
        
        # Загружаем модель эмбеддингов
        embedding_model = self.load_embedding_model()
        
        # Загружаем документы
        documents_json_path = os.path.join(dataset_path, "documents_for_rag.json")
        documents = self.load_documents_from_json(documents_json_path)
        
        if not documents:
            raise ValueError("Не найдено документов для загрузки")
        
        # Создаем векторную БД
        vector_store_type = self.config['vector_store']['type']
        
        if vector_store_type == 'chroma':
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=vector_store_path
            )
        elif vector_store_type == 'faiss':
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embedding_model
            )
            vector_store.save_local(vector_store_path)
        else:
            raise ValueError(f"Неподдерживаемый тип векторной БД: {vector_store_type}")
        
        logger.info(f"Векторная БД создана и сохранена в {vector_store_path}")
        
        return vector_store
    
    def load_vector_store_from_dataset(self, dataset_path: str, vector_store_path: str) -> Any:
        """
        Загружает существующую векторную БД из датасета.
        
        Args:
            dataset_path: Путь к папке с датасетом
            vector_store_path: Путь к векторной БД
            
        Returns:
            Векторная БД
        """
        logger.info(f"Загрузка векторной БД из {vector_store_path}")
        
        # Загружаем модель эмбеддингов
        embedding_model = self.load_embedding_model()
        
        # Загружаем векторную БД
        vector_store_type = self.config['vector_store']['type']
        
        if vector_store_type == 'chroma':
            vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embedding_model
            )
        elif vector_store_type == 'faiss':
            vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
        else:
            raise ValueError(f"Неподдерживаемый тип векторной БД: {vector_store_type}")
        
        logger.info("Векторная БД загружена")
        return vector_store
    
    def get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """
        Получает информацию о датасете.
        
        Args:
            dataset_path: Путь к папке с датасетом
            
        Returns:
            Dict[str, Any]: Информация о датасете
        """
        info = {
            'dataset_path': dataset_path,
            'documents_count': 0,
            'qa_pairs_count': 0,
            'documents_file': None,
            'qa_pairs_file': None
        }
        
        # Проверяем файл с документами
        documents_file = os.path.join(dataset_path, "documents_for_rag.json")
        if os.path.exists(documents_file):
            with open(documents_file, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            info['documents_count'] = len(documents_data)
            info['documents_file'] = documents_file
        
        # Проверяем файл с парами вопрос-ответ
        qa_pairs_file = os.path.join(dataset_path, "qa_pairs.json")
        if os.path.exists(qa_pairs_file):
            with open(qa_pairs_file, 'r', encoding='utf-8') as f:
                qa_pairs_data = json.load(f)
            info['qa_pairs_count'] = len(qa_pairs_data)
            info['qa_pairs_file'] = qa_pairs_file
        
        return info


def create_dataset_loader(config: Dict[str, Any]) -> DatasetLoader:
    """Создает экземпляр DatasetLoader"""
    return DatasetLoader(config)
