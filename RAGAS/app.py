"""
Веб-интерфейс для RAG системы.
Поддерживает как Gradio, так и Streamlit интерфейсы.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import RAGPipeline, create_rag_pipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGWebApp:
    """Класс для веб-интерфейса RAG системы."""
    
    def __init__(self, config_path: str):
        """
        Инициализация веб-приложения.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.pipeline = None
        self.web_config = self.config.get('web', {})
        
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из YAML файла."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Конфигурация загружена из: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
            raise
    
    def initialize_pipeline(self) -> None:
        """Инициализирует RAG пайплайн."""
        try:
            logger.info("Инициализация RAG пайплайна...")
            self.pipeline = create_rag_pipeline(self.config)
            self.pipeline.initialize()
            logger.info("RAG пайплайн инициализирован успешно")
        except Exception as e:
            logger.error(f"Ошибка при инициализации пайплайна: {e}")
            raise
    
    def query_pipeline(self, question: str) -> Dict[str, Any]:
        """
        Выполняет запрос к пайплайну.
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Dict[str, Any]: Ответ и метаданные
        """
        if not self.pipeline:
            return {
                "answer": "Ошибка: Пайплайн не инициализирован",
                "source_documents": [],
                "error": "Pipeline not initialized"
            }
        
        try:
            return self.pipeline.query(question, return_sources=True)
        except Exception as e:
            logger.error(f"Ошибка при выполнении запроса: {e}")
            return {
                "answer": f"Ошибка при обработке запроса: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Возвращает информацию о пайплайне."""
        if not self.pipeline:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "vector_store_info": self.pipeline.get_vector_store_info(),
            "stats": self.pipeline.get_stats(),
            "model_info": {
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "generator_model": "rag",
                "device": "cuda"
            }
        }


def create_gradio_interface(app: RAGWebApp):
    """Создает Gradio интерфейс."""
    import gradio as gr
    
    def process_query(question: str, history: List[List[str]]) -> tuple:
        """Обрабатывает запрос пользователя."""
        if not question.strip():
            return history, "", ""
        
        # Выполняем запрос
        result = app.query_pipeline(question)
        
        # Формируем ответ
        answer = result.get("answer", "Нет ответа")
        response_time = result.get("response_time", 0)
        
        # Формируем информацию об источниках
        sources_text = ""
        if result.get("source_documents"):
            sources_text = "**Источники:**\n\n"
            for i, doc in enumerate(result["source_documents"], 1):
                content = doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "Неизвестный источник")
                sources_text += f"{i}. **{source}**\n{content}\n\n"
        
        # Обновляем историю
        history.append([question, answer])
        
        # Добавляем информацию о времени ответа
        info_text = f"Время ответа: {response_time:.2f} сек"
        
        return history, sources_text, info_text
    
    def get_system_info():
        """Возвращает информацию о системе."""
        info = app.get_pipeline_info()
        
        if info["status"] == "not_initialized":
            return "Система не инициализирована"
        
        vector_info = info["vector_store_info"]
        stats = info["stats"]
        
        info_text = f"""
**Информация о системе:**
- Статус: {info["status"]}
- Тип векторной базы: {vector_info.get("type", "Неизвестно")}
- Количество документов: {vector_info.get("document_count", "Неизвестно")}
- Всего запросов: {stats.get("total_queries", 0)}
- Среднее время ответа: {stats.get("avg_response_time", 0):.2f} сек
        """
        
        return info_text.strip()
    
    # Создаем интерфейс
    with gr.Blocks(title="RAG Система", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 RAG Система для Вопросов и Ответов")
        gr.Markdown("Задайте вопрос, и система найдет релевантную информацию и сгенерирует ответ.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Чат интерфейс
                chatbot = gr.Chatbot(
                    label="Диалог",
                    height=400,
                    show_label=True
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Введите ваш вопрос...",
                        label="Вопрос",
                        lines=2
                    )
                    submit_btn = gr.Button("Отправить", variant="primary")
                
                # Информация о времени ответа
                response_info = gr.Textbox(
                    label="Информация о запросе",
                    interactive=False,
                    lines=1
                )
            
            with gr.Column(scale=1):
                # Информация о системе
                system_info = gr.Textbox(
                    label="Информация о системе",
                    value=get_system_info(),
                    interactive=False,
                    lines=10
                )
                
                # Обновить информацию
                refresh_btn = gr.Button("Обновить информацию")
                
                # Источники
                sources_output = gr.Textbox(
                    label="Источники информации",
                    interactive=False,
                    lines=15
                )
        
        # Обработчики событий
        def submit_question(question, history):
            return process_query(question, history)
        
        submit_btn.click(
            fn=submit_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, sources_output, response_info]
        )
        
        question_input.submit(
            fn=submit_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, sources_output, response_info]
        )
        
        refresh_btn.click(
            fn=get_system_info,
            outputs=[system_info]
        )
        
        # Примеры вопросов
        gr.Markdown("### 💡 Примеры вопросов:")
        examples = gr.Examples(
            examples=[
                "Что такое машинное обучение?",
                "Какие типы нейронных сетей существуют?",
                "Как работает алгоритм градиентного спуска?",
                "Что такое глубокое обучение?",
                "Какие метрики используются для оценки моделей?"
            ],
            inputs=[question_input]
        )
    
    return interface


def create_streamlit_interface(app: RAGWebApp):
    """Создает Streamlit интерфейс."""
    import streamlit as st
    
    # Настройка страницы
    st.set_page_config(
        page_title="RAG Система",
        page_icon="🤖",
        layout="wide"
    )
    
    # Заголовок
    st.title("🤖 RAG Система для Вопросов и Ответов")
    st.markdown("Задайте вопрос, и система найдет релевантную информацию и сгенерирует ответ.")
    
    # Боковая панель с информацией о системе
    with st.sidebar:
        st.header("📊 Информация о системе")
        
        info = app.get_pipeline_info()
        
        if info["status"] == "not_initialized":
            st.error("Система не инициализирована")
        else:
            vector_info = info["vector_store_info"]
            stats = info["stats"]
            
            st.metric("Статус", info["status"])
            st.metric("Тип БД", vector_info.get("type", "Неизвестно"))
            st.metric("Документов", vector_info.get("document_count", "Неизвестно"))
            st.metric("Запросов", stats.get("total_queries", 0))
            st.metric("Ср. время ответа", f"{stats.get('avg_response_time', 0):.2f} сек")
            
            if st.button("🔄 Обновить информацию"):
                st.rerun()
    
    # Основной интерфейс
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Поле для ввода вопроса
        question = st.text_area(
            "Введите ваш вопрос:",
            placeholder="Например: Что такое машинное обучение?",
            height=100
        )
        
        # Кнопка отправки
        if st.button("🔍 Найти ответ", type="primary"):
            if question.strip():
                with st.spinner("Обрабатываем ваш запрос..."):
                    result = app.query_pipeline(question)
                    
                    # Отображаем ответ
                    st.subheader("💬 Ответ:")
                    st.write(result.get("answer", "Нет ответа"))
                    
                    # Информация о времени ответа
                    response_time = result.get("response_time", 0)
                    st.caption(f"⏱️ Время ответа: {response_time:.2f} секунд")
                    
                    # Отображаем источники
                    if result.get("source_documents"):
                        st.subheader("📚 Источники информации:")
                        for i, doc in enumerate(result["source_documents"], 1):
                            with st.expander(f"Источник {i}"):
                                content = doc.get("content", "")
                                metadata = doc.get("metadata", {})
                                source = metadata.get("source", "Неизвестный источник")
                                
                                st.write(f"**Файл:** {source}")
                                st.write(f"**Содержимое:** {content}")
                                
                                if "score" in doc and doc["score"] is not None:
                                    st.write(f"**Релевантность:** {doc['score']:.4f}")
            else:
                st.warning("Пожалуйста, введите вопрос")
    
    with col2:
        # Примеры вопросов
        st.subheader("💡 Примеры вопросов:")
        
        example_questions = [
            "Что такое машинное обучение?",
            "Какие типы нейронных сетей существуют?",
            "Как работает алгоритм градиентного спуска?",
            "Что такое глубокое обучение?",
            "Какие метрики используются для оценки моделей?"
        ]
        
        for i, example in enumerate(example_questions, 1):
            if st.button(f"{i}. {example}", key=f"example_{i}"):
                st.session_state.example_question = example
                st.rerun()
        
        # Обработка выбранного примера
        if "example_question" in st.session_state:
            question = st.session_state.example_question
            del st.session_state.example_question
            st.rerun()
    
    # Футер
    st.markdown("---")
    st.markdown("**RAG Система** - Retrieval-Augmented Generation для вопросов и ответов")


def main():
    """Основная функция для запуска веб-приложения."""
    parser = argparse.ArgumentParser(description='RAG Web Application')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--interface', type=str, choices=['gradio', 'streamlit'],
                       help='Тип интерфейса (переопределяет конфигурацию)')
    parser.add_argument('--port', type=int, help='Порт для запуска')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Хост для запуска')
    parser.add_argument('--share', action='store_true', help='Поделиться интерфейсом (только для Gradio)')
    
    args = parser.parse_args()
    
    try:
        # Создаем приложение
        app = RAGWebApp(args.config)
        
        # Инициализируем пайплайн
        app.initialize_pipeline()
        
        # Определяем тип интерфейса
        interface_type = args.interface or app.web_config.get('interface', 'gradio')
        
        if interface_type == 'gradio':
            # Запускаем Gradio интерфейс
            import gradio as gr
            
            interface = create_gradio_interface(app)
            
            # Параметры запуска
            port = args.port or app.web_config.get('port', 7860)
            host = args.host or app.web_config.get('host', '0.0.0.0')
            share = args.share or app.web_config.get('share', False)
            
            logger.info(f"Запуск Gradio интерфейса на {host}:{port}")
            interface.launch(
                server_name=host,
                server_port=port,
                share=share,
                show_error=True
            )
            
        elif interface_type == 'streamlit':
            # Запускаем Streamlit интерфейс
            import streamlit as st
            
            # Создаем интерфейс
            create_streamlit_interface(app)
            
            # Параметры запуска
            port = args.port or app.web_config.get('port', 8501)
            host = args.host or app.web_config.get('host', '0.0.0.0')
            
            logger.info(f"Запуск Streamlit интерфейса на {host}:{port}")
            
            # Streamlit запускается через команду streamlit run
            print(f"Для запуска Streamlit интерфейса выполните:")
            print(f"streamlit run app.py -- --config {args.config} --interface streamlit")
    
    except Exception as e:
        logger.error(f"Ошибка при запуске веб-приложения: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

