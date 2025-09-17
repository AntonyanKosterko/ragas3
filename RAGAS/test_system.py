#!/usr/bin/env python3
"""
Простой тест для проверки работы RAG системы.
"""

import sys
import os
sys.path.append('src')

from src.pipeline import create_rag_pipeline
import yaml

def test_rag_system():
    """Тестирует базовую функциональность RAG системы."""
    print("🧪 Тестирование RAG системы...")
    
    # Загружаем конфигурацию
    with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Создаем пайплайн
    print("📦 Создание RAG пайплайна...")
    pipeline = create_rag_pipeline(config)
    pipeline.initialize()
    
    # Тестовые вопросы
    test_questions = [
        "Что такое машинное обучение?",
        "Какие типы нейронных сетей существуют?",
        "Как работает градиентный спуск?"
    ]
    
    print("\n🔍 Тестирование вопросов:")
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Вопрос: {question}")
        try:
            result = pipeline.query(question)
            print(f"   Ответ: {result['answer'][:200]}...")
            print(f"   Время: {result['response_time']:.2f} сек")
            print(f"   Источников: {len(result.get('source_documents', []))}")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    # Статистика
    stats = pipeline.get_stats()
    print(f"\n📊 Статистика пайплайна:")
    print(f"   Всего запросов: {stats['total_queries']}")
    print(f"   Среднее время ответа: {stats['avg_response_time']:.2f} сек")
    
    print("\n✅ Тест завершен успешно!")

if __name__ == "__main__":
    test_rag_system()

