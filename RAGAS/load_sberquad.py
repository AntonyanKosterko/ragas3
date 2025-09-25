#!/usr/bin/env python3
"""
Загрузчик датасета SberQuAD для RAG системы
"""

import os
import json
import logging
from datasets import load_dataset
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_sberquad_dataset():
    """Загружает датасет SberQuAD с Hugging Face"""
    print("📥 Загрузка датасета SberQuAD...")
    
    try:
        # Загружаем датасет
        dataset = load_dataset("kuznetsoffandrey/sberquad")
        
        # Создаем директорию для датасета
        os.makedirs("datasets/sberquad", exist_ok=True)
        
        # Сохраняем train split
        train_data = dataset['train']
        train_data.to_json("datasets/sberquad/train.json")
        
        print(f"✅ Датасет загружен: {len(train_data)} примеров")
        print(f"📁 Сохранено в: datasets/sberquad/")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке датасета: {e}")
        return None


def create_documents_from_sberquad(dataset, max_documents=500):
    """Создает документы для RAG из датасета SberQuAD"""
    print(f"📄 Создание документов для RAG (максимум {max_documents})...")
    
    # Извлекаем уникальные контексты
    contexts = {}
    processed = 0
    
    for item in dataset['train']:
        if len(contexts) >= max_documents:
            break
            
        context = item['context']
        context_id = f"context_{hash(context) % 10000}"
        
        if context_id not in contexts:
            contexts[context_id] = {
                'id': context_id,
                'title': f"Контекст {context_id}",
                'content': context,
                'source': 'sberquad',
                'type': 'context'
            }
        
        processed += 1
        if processed % 1000 == 0:
            print(f"  Обработано {processed} примеров, найдено {len(contexts)} уникальных контекстов...")
    
    # Сохраняем документы
    documents = list(contexts.values())
    with open("datasets/sberquad/documents_for_rag.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Создано {len(documents)} документов из {processed} обработанных примеров")
    return documents


def create_qa_pairs_from_sberquad(dataset, max_samples=None):
    """Создает пары вопрос-ответ из датасета SberQuAD"""
    print("❓ Создание пар вопрос-ответ...")
    
    qa_pairs = []
    train_data = dataset['train']
    
    # Ограничиваем количество примеров если нужно
    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
    
    for i, item in enumerate(train_data):
        qa_pair = {
            'id': f"qa_{i}",
            'question': item['question'],
            'answer': item['answers']['text'][0] if item['answers']['text'] else "",
            'context_doc_ids': [f"context_{hash(item['context']) % 10000}"],
            'difficulty': 'medium'  # SberQuAD не имеет метки сложности
        }
        qa_pairs.append(qa_pair)
    
    # Сохраняем QA пары
    with open("datasets/sberquad/qa_pairs.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Создано {len(qa_pairs)} пар вопрос-ответ")
    return qa_pairs


def analyze_sberquad_dataset(dataset):
    """Анализирует датасет SberQuAD"""
    print("\n📊 Анализ датасета SberQuAD:")
    print("=" * 50)
    
    train_data = dataset['train']
    print(f"Общее количество примеров: {len(train_data)}")
    
    # Анализ длины вопросов и ответов
    question_lengths = [len(item['question']) for item in train_data]
    answer_lengths = [len(item['answers']['text'][0]) if item['answers']['text'] else 0 for item in train_data]
    context_lengths = [len(item['context']) for item in train_data]
    
    print(f"Вопросы - мин: {min(question_lengths)}, макс: {max(question_lengths)}, среднее: {sum(question_lengths)/len(question_lengths):.1f}")
    print(f"Ответы - мин: {min(answer_lengths)}, макс: {max(answer_lengths)}, среднее: {sum(answer_lengths)/len(answer_lengths):.1f}")
    print(f"Контексты - мин: {min(context_lengths)}, макс: {max(context_lengths)}, среднее: {sum(context_lengths)/len(context_lengths):.1f}")
    
    # Примеры
    print("\n📝 Примеры из датасета:")
    for i in range(min(3, len(train_data))):
        item = train_data[i]
        print(f"\n--- Пример {i+1} ---")
        print(f"Вопрос: {item['question']}")
        print(f"Ответ: {item['answers']['text'][0] if item['answers']['text'] else 'Нет ответа'}")
        print(f"Контекст: {item['context'][:200]}...")


def main():
    """Основная функция для загрузки и подготовки датасета SberQuAD"""
    print("🚀 Подготовка датасета SberQuAD для RAG системы")
    print("=" * 60)
    
    # Загружаем датасет
    dataset = load_sberquad_dataset()
    if not dataset:
        return
    
    # Анализируем датасет
    analyze_sberquad_dataset(dataset)
    
    # Создаем документы для RAG (увеличиваем до 2000 для демонстрации прогресс-бара)
    documents = create_documents_from_sberquad(dataset, max_documents=2000)
    
    # Создаем пары вопрос-ответ (увеличиваем до 500 для тестирования)
    qa_pairs = create_qa_pairs_from_sberquad(dataset, max_samples=500)
    
    print("\n✅ Датасет SberQuAD готов для использования в RAG системе!")
    print(f"📄 Документов: {len(documents)}")
    print(f"❓ Вопросов-ответов: {len(qa_pairs)}")
    print("📁 Все файлы сохранены в: datasets/sberquad/")


if __name__ == "__main__":
    main()
