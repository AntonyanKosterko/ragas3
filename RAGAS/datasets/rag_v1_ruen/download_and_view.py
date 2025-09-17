"""
Скрипт для скачивания и просмотра датасета RAG-v1-ruen
"""

import os
import json
from datasets import load_dataset
from typing import List, Dict, Any

def download_dataset():
    """Скачивает датасет RAG-v1-ruen с Hugging Face"""
    print("Скачивание датасета RAG-v1-ruen...")
    
    try:
        # Загружаем датасет
        dataset = load_dataset("MexIvanov/RAG-v1-ruen")
        
        # Сохраняем в локальные файлы
        output_dir = "/home/antonkosterin/Courses/RAG/RAGAS/datasets/rag_v1_ruen"
        
        # Сохраняем train split
        train_data = dataset['train']
        train_data.to_json(f"{output_dir}/train.json")
        
        print(f"Датасет сохранен в {output_dir}")
        print(f"Размер train split: {len(train_data)} примеров")
        
        return dataset
        
    except Exception as e:
        print(f"Ошибка при скачивании датасета: {e}")
        return None

def view_dataset_samples(dataset, num_samples: int = 10):
    """Показывает примеры из датасета"""
    print(f"\n=== Примеры из датасета RAG-v1-ruen (первые {num_samples}) ===")
    
    train_data = dataset['train']
    
    for i in range(min(num_samples, len(train_data))):
        sample = train_data[i]
        print(f"\n--- Пример {i+1} ---")
        print(f"Вопрос: {sample.get('question', 'N/A')}")
        print(f"Ответ: {sample.get('answer', 'N/A')}")
        print(f"Контекст: {sample.get('context', 'N/A')[:200]}...")
        print(f"Источник: {sample.get('source', 'N/A')}")
        print("-" * 50)

def analyze_dataset(dataset):
    """Анализирует датасет и выводит статистику"""
    print("\n=== Анализ датасета RAG-v1-ruen ===")
    
    train_data = dataset['train']
    
    # Общая статистика
    print(f"Общее количество примеров: {len(train_data)}")
    
    # Анализ языков
    languages = {}
    for sample in train_data:
        lang = sample.get('language', 'unknown')
        languages[lang] = languages.get(lang, 0) + 1
    
    print(f"\nРаспределение по языкам:")
    for lang, count in languages.items():
        print(f"  {lang}: {count} ({count/len(train_data)*100:.1f}%)")
    
    # Анализ источников
    sources = {}
    for sample in train_data:
        source = sample.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print(f"\nТоп-10 источников:")
    sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
    for source, count in sorted_sources[:10]:
        print(f"  {source}: {count}")
    
    # Анализ длины вопросов и ответов
    question_lengths = [len(sample.get('question', '')) for sample in train_data]
    answer_lengths = [len(sample.get('answer', '')) for sample in train_data]
    
    print(f"\nСтатистика длины:")
    print(f"  Вопросы - мин: {min(question_lengths)}, макс: {max(question_lengths)}, среднее: {sum(question_lengths)/len(question_lengths):.1f}")
    print(f"  Ответы - мин: {min(answer_lengths)}, макс: {max(answer_lengths)}, среднее: {sum(answer_lengths)/len(answer_lengths):.1f}")

def extract_documents_for_rag(dataset, output_file: str = "documents_for_rag.json"):
    """Извлекает уникальные документы из датасета для загрузки в векторную БД"""
    print("\n=== Извлечение документов для RAG ===")
    
    train_data = dataset['train']
    documents = {}
    
    # Собираем уникальные документы по источникам
    for sample in train_data:
        source = sample.get('source', 'unknown')
        context = sample.get('context', '')
        
        if source not in documents and context:
            documents[source] = {
                'content': context,
                'metadata': {
                    'source': source,
                    'language': sample.get('language', 'unknown'),
                    'type': 'rag_document'
                }
            }
    
    # Сохраняем документы
    output_path = f"/home/antonkosterin/Courses/RAG/RAGAS/datasets/rag_v1_ruen/{output_file}"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Извлечено {len(documents)} уникальных документов")
    print(f"Документы сохранены в {output_path}")
    
    return documents

def main():
    """Основная функция"""
    print("=== RAG-v1-ruen Dataset Viewer ===")
    
    # Скачиваем датасет
    dataset = download_dataset()
    if dataset is None:
        return
    
    # Показываем примеры
    view_dataset_samples(dataset, 5)
    
    # Анализируем датасет
    analyze_dataset(dataset)
    
    # Извлекаем документы для RAG
    documents = extract_documents_for_rag(dataset)
    
    print("\n=== Готово! ===")
    print("Датасет готов для использования в RAG системе")

if __name__ == "__main__":
    main()
