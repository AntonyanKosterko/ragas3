"""
Простая генеративная модель для RAG системы.
Использует только ретривенные документы без сложной генерации.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

logger = logging.getLogger(__name__)


class SimpleGenerator(BaseLanguageModel):
    """Простая генеративная модель, которая формирует ответ на основе ретривенных документов."""
    
    def __init__(self):
        super().__init__()
        self.name = "simple_generator"
        logger.info("Простая генеративная модель инициализирована")
    
    @property
    def _llm_type(self) -> str:
        """Возвращает тип модели."""
        return "simple_generator"
    
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
            # Простая обработка промпта
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
                context_docs = [{"content": context}]
                return self.generate(query, context_docs)
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
        lines = prompt.strip().split('\n')
        context = ""
        question = ""
        
        in_context = False
        in_question = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("Контекст:"):
                in_context = True
                in_question = False
                context = line.replace("Контекст:", "").strip()
            elif line.startswith("Вопрос:"):
                in_question = True
                in_context = False
                question = line.replace("Вопрос:", "").strip()
            elif in_context and line:
                context += "\n" + line
            elif in_question and line:
                question += "\n" + line
        
        # Формируем ответ на основе контекста
        if context and question:
            context_docs = [{"content": context.strip()}]
            return self.generate(question.strip(), context_docs)
        else:
            # Если не удалось извлечь контекст, возвращаем общий ответ
            return "Основываясь на доступной информации, я могу помочь вам с вопросами о машинном обучении, нейронных сетях, алгоритмах оптимизации и других темах, связанных с искусственным интеллектом."
    
    def generate(self, question: str, context_documents: List[Dict[str, Any]]) -> str:
        """
        Генерирует ответ на основе вопроса и контекстных документов.
        
        Args:
            question: Вопрос пользователя
            context_documents: Список ретривенных документов
            
        Returns:
            str: Сгенерированный ответ
        """
        if not context_documents:
            return "Извините, я не смог найти релевантную информацию для ответа на ваш вопрос."
        
        # Формируем ответ на основе контекста
        answer_parts = []
        
        # Добавляем введение
        answer_parts.append(f"Основываясь на найденной информации, вот ответ на ваш вопрос: '{question}'\n\n")
        
        # Добавляем информацию из документов
        for i, doc in enumerate(context_documents, 1):
            content = doc.get('content', '')
            if content:
                # Обрезаем длинный контент
                if len(content) > 500:
                    content = content[:500] + "..."
                
                answer_parts.append(f"**Источник {i}:**\n{content}\n\n")
        
        # Добавляем заключение
        answer_parts.append("Это основная информация, которую я смог найти по вашему вопросу.")
        
        return "".join(answer_parts)
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Вызов модели в стиле LangChain.
        
        Args:
            inputs: Словарь с ключами 'query' и 'context'
            
        Returns:
            Dict[str, Any]: Результат генерации
        """
        question = inputs.get('query', '')
        context = inputs.get('context', '')
        
        # Парсим контекст из строки (простая реализация)
        context_documents = []
        if context:
            # Разбиваем контекст на части (упрощенная версия)
            context_parts = context.split('\n\n')
            for part in context_parts:
                if part.strip():
                    context_documents.append({'content': part.strip()})
        
        answer = self.generate(question, context_documents)
        
        return {
            'result': answer,
            'source_documents': context_documents
        }
