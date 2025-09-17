"""
Скрипт для скачивания и просмотра датасета RAG Bench от AI Forever
"""

import os
import json
from datasets import load_dataset
from typing import List, Dict, Any

def download_rag_bench():
    """Скачивает датасеты RAG Bench с Hugging Face"""
    print("Скачивание датасетов RAG Bench...")
    
    try:
        # Загружаем тексты
        texts_dataset = load_dataset("ai-forever/rag-bench-public-texts")
        
        # Загружаем вопросы
        questions_dataset = load_dataset("ai-forever/rag-bench-public-questions")
        
        # Сохраняем в локальные файлы
        output_dir = "/home/antonkosterin/Courses/RAG/RAGAS/datasets/rag_bench"
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем тексты
        texts_data = texts_dataset['train']
        texts_data.to_json(f"{output_dir}/texts.json")
        
        # Сохраняем вопросы
        questions_data = questions_dataset['train']
        questions_data.to_json(f"{output_dir}/questions.json")
        
        print(f"Датасеты сохранены в {output_dir}")
        print(f"Размер текстов: {len(texts_data)}")
        print(f"Размер вопросов: {len(questions_data)}")
        
        return texts_dataset, questions_dataset
        
    except Exception as e:
        print(f"Ошибка при скачивании датасетов: {e}")
        return None, None

def view_dataset_samples(texts_dataset, questions_dataset, num_samples: int = 5):
    """Показывает примеры из датасетов"""
    print(f"\n=== Примеры из RAG Bench (первые {num_samples}) ===")
    
    texts_data = texts_dataset['train']
    questions_data = questions_dataset['train']
    
    print("\n--- Тексты ---")
    for i in range(min(num_samples, len(texts_data))):
        sample = texts_data[i]
        print(f"\nТекст {i+1}:")
        print(f"ID: {sample.get('id', 'N/A')}")
        print(f"Содержимое: {sample.get('text', 'N/A')[:300]}...")
        print(f"Источник: {sample.get('source', 'N/A')}")
        print("-" * 50)
    
    print("\n--- Вопросы ---")
    for i in range(min(num_samples, len(questions_data))):
        sample = questions_data[i]
        print(f"\nВопрос {i+1}:")
        print(f"ID: {sample.get('id', 'N/A')}")
        print(f"Вопрос: {sample.get('question', 'N/A')}")
        print(f"Ответ: {sample.get('answer', 'N/A')}")
        print(f"Источник: {sample.get('source', 'N/A')}")
        print("-" * 50)

def analyze_datasets(texts_dataset, questions_dataset):
    """Анализирует датасеты и выводит статистику"""
    print("\n=== Анализ датасетов RAG Bench ===")
    
    texts_data = texts_dataset['train']
    questions_data = questions_dataset['train']
    
    print(f"Количество текстов: {len(texts_data)}")
    print(f"Количество вопросов: {len(questions_data)}")
    
    # Анализ текстов
    text_lengths = [len(sample.get('text', '')) for sample in texts_data]
    print(f"\nСтатистика текстов:")
    print(f"  Длина - мин: {min(text_lengths)}, макс: {max(text_lengths)}, среднее: {sum(text_lengths)/len(text_lengths):.1f}")
    
    # Анализ вопросов
    question_lengths = [len(sample.get('question', '')) for sample in questions_data]
    answer_lengths = [len(sample.get('answer', '')) for sample in questions_data]
    
    print(f"\nСтатистика вопросов:")
    print(f"  Вопросы - мин: {min(question_lengths)}, макс: {max(question_lengths)}, среднее: {sum(question_lengths)/len(question_lengths):.1f}")
    print(f"  Ответы - мин: {min(answer_lengths)}, макс: {max(answer_lengths)}, среднее: {sum(answer_lengths)/len(answer_lengths):.1f}")

def extract_documents_for_rag(texts_dataset, output_file: str = "documents_for_rag.json"):
    """Извлекает документы из датасета для загрузки в векторную БД"""
    print("\n=== Извлечение документов для RAG ===")
    
    texts_data = texts_dataset['train']
    documents = {}
    
    # Собираем документы
    for i, sample in enumerate(texts_data):
        doc_id = sample.get('id', f'doc_{i}')
        text = sample.get('text', '')
        source = sample.get('source', 'unknown')
        
        if text:
            documents[doc_id] = {
                'content': text,
                'metadata': {
                    'source': source,
                    'id': doc_id,
                    'type': 'rag_bench_document'
                }
            }
    
    # Сохраняем документы
    output_path = f"/home/antonkosterin/Courses/RAG/RAGAS/datasets/rag_bench/{output_file}"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Извлечено {len(documents)} документов")
    print(f"Документы сохранены в {output_path}")
    
    return documents

def extract_qa_pairs(questions_dataset, output_file: str = "qa_pairs.json"):
    """Извлекает пары вопрос-ответ для тестирования RAG"""
    print("\n=== Извлечение пар вопрос-ответ для тестирования ===")
    
    questions_data = questions_dataset['train']
    qa_pairs = []
    
    # Собираем пары вопрос-ответ
    for i, sample in enumerate(questions_data):
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        source = sample.get('source', 'unknown')
        
        if question and answer:
            qa_pairs.append({
                'id': sample.get('id', f'qa_{i}'),
                'question': question,
                'answer': answer,
                'source': source,
                'metadata': {
                    'type': 'rag_bench_qa'
                }
            })
    
    # Сохраняем пары
    output_path = f"/home/antonkosterin/Courses/RAG/RAGAS/datasets/rag_bench/{output_file}"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"Извлечено {len(qa_pairs)} пар вопрос-ответ")
    print(f"Пары сохранены в {output_path}")
    
    return qa_pairs

def main():
    """Основная функция"""
    print("=== RAG Bench Dataset Viewer ===")
    
    # Скачиваем датасеты
    texts_dataset, questions_dataset = download_rag_bench()
    if texts_dataset is None or questions_dataset is None:
        return
    
    # Показываем примеры
    view_dataset_samples(texts_dataset, questions_dataset, 3)
    
    # Анализируем датасеты
    analyze_datasets(texts_dataset, questions_dataset)
    
    # Извлекаем документы для RAG
    documents = extract_documents_for_rag(texts_dataset)
    
    # Извлекаем пары вопрос-ответ
    qa_pairs = extract_qa_pairs(questions_dataset)
    
    print("\n=== Готово! ===")
    print("Датасеты готовы для использования в RAG системе")

if __name__ == "__main__":
    main()
