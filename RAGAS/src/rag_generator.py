"""
Генеративная модель специально для RAG системы.
Извлекает релевантную информацию из контекста и формирует ответ.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

logger = logging.getLogger(__name__)


class RAGGenerator(BaseLanguageModel):
    """Генеративная модель для RAG системы, которая извлекает релевантную информацию из контекста."""
    
    def __init__(self):
        super().__init__()
        self.name = "rag_generator"
        logger.info("RAG генеративная модель инициализирована")
    
    @property
    def _llm_type(self) -> str:
        """Возвращает тип модели."""
        return "rag_generator"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Генерирует ответы для списка промптов."""
        generations = []
        for prompt in prompts:
            response = self._process_prompt(prompt)
            generations.append([Generation(text=response)])
        
        return LLMResult(generations=generations)
    
    def invoke(self, input: Union[str, Dict[str, Any]], **kwargs: Any) -> str:
        """Вызывает модель с входными данными."""
        if isinstance(input, str):
            return self._process_prompt(input)
        elif isinstance(input, dict):
            query = input.get("query", "")
            context = input.get("context", "")
            if context:
                return self._generate_answer(query, context)
            else:
                return self._process_prompt(query)
        else:
            return "Неверный формат входных данных"
    
    def predict(self, text: str, **kwargs: Any) -> str:
        """Предсказывает следующий текст."""
        return self._process_prompt(text)
    
    def predict_messages(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """Предсказывает ответ на сообщения."""
        if messages:
            last_message = messages[-1]
            content = last_message.get("content", "")
            return self._process_prompt(content)
        return "Нет сообщений для обработки"
    
    def generate_prompt(self, prompts: List[Dict[str, Any]], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        """Генерирует ответы для промптов."""
        text_prompts = []
        for prompt in prompts:
            if isinstance(prompt, dict):
                text_prompts.append(prompt.get("text", ""))
            else:
                text_prompts.append(str(prompt))
        
        return self._generate(text_prompts, stop=stop, **kwargs)
    
    def agenerate_prompt(self, prompts: List[Dict[str, Any]], **kwargs: Any) -> LLMResult:
        """Асинхронная генерация ответов для промптов."""
        return self.generate_prompt(prompts, **kwargs)
    
    def apredict(self, text: str, **kwargs: Any) -> str:
        """Асинхронное предсказание."""
        return self.predict(text, **kwargs)
    
    def apredict_messages(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """Асинхронное предсказание для сообщений."""
        return self.predict_messages(messages, **kwargs)
    
    def _process_prompt(self, prompt: str) -> str:
        """Обрабатывает промпт и возвращает ответ."""
        # Извлекаем контекст и вопрос из промпта
        context_match = re.search(r'Контекст:\s*(.*?)(?=Вопрос:|$)', prompt, re.DOTALL)
        question_match = re.search(r'Вопрос:\s*(.*?)(?=Ответ|$)', prompt, re.DOTALL)
        
        if context_match and question_match:
            context = context_match.group(1).strip()
            question = question_match.group(1).strip()
            return self._generate_answer(question, context)
        else:
            return "Извините, я не смог извлечь контекст и вопрос из запроса."
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Генерирует ответ на основе вопроса и контекста."""
        # Простой анализ вопроса для поиска релевантной информации
        question_lower = question.lower()
        
        # Ищем релевантные части контекста
        relevant_parts = []
        context_paragraphs = context.split('\n\n')
        
        for paragraph in context_paragraphs:
            if self._is_relevant(question_lower, paragraph.lower()):
                relevant_parts.append(paragraph.strip())
        
        if not relevant_parts:
            return "Извините, я не смог найти релевантную информацию для ответа на ваш вопрос в предоставленном контексте."
        
        # Формируем ответ
        answer_parts = []
        
        # Добавляем введение
        if "что такое" in question_lower:
            answer_parts.append("Основываясь на предоставленной информации:\n\n")
        elif "какие" in question_lower or "как" in question_lower:
            answer_parts.append("Согласно информации из контекста:\n\n")
        else:
            answer_parts.append("Вот информация по вашему вопросу:\n\n")
        
        # Добавляем релевантные части (максимум 3)
        for i, part in enumerate(relevant_parts[:3], 1):
            if len(part) > 500:
                part = part[:500] + "..."
            answer_parts.append(f"{i}. {part}\n")
        
        return "".join(answer_parts)
    
    def _is_relevant(self, question: str, paragraph: str) -> bool:
        """Определяет, релевантен ли параграф для вопроса."""
        # Ключевые слова для разных типов вопросов
        keywords_map = {
            "машинное обучение": ["машинное обучение", "machine learning", "ml", "алгоритм", "модель"],
            "нейронные сети": ["нейронные сети", "neural networks", "глубокое обучение", "deep learning", "cnn", "rnn", "трансформер"],
            "градиентный спуск": ["градиентный спуск", "gradient descent", "оптимизация", "sgd", "adam", "rmsprop"],
            "обучение": ["обучение", "training", "supervised", "unsupervised", "reinforcement"],
            "метрики": ["метрики", "accuracy", "precision", "recall", "f1", "mse", "mae"],
            "применения": ["применения", "приложения", "области", "компьютерное зрение", "nlp", "рекомендательные системы"]
        }
        
        # Ищем ключевые слова в вопросе
        for topic, keywords in keywords_map.items():
            if any(keyword in question for keyword in [topic] + keywords):
                # Проверяем, есть ли эти ключевые слова в параграфе
                if any(keyword in paragraph for keyword in keywords):
                    return True
        
        # Если не нашли специфические ключевые слова, проверяем общую релевантность
        question_words = set(question.split())
        paragraph_words = set(paragraph.split())
        
        # Если есть пересечение слов (минимум 2 общих слова)
        common_words = question_words.intersection(paragraph_words)
        return len(common_words) >= 2
