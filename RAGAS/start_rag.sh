#!/bin/bash

# Скрипт для запуска RAG системы с полноценным датасетом
# Автор: RAG Assistant
# Дата: $(date)

echo "🚀 Запуск RAG системы с полноценным датасетом"
echo "=============================================="

# Проверяем, что мы в правильной директории
if [ ! -f "app.py" ]; then
    echo "❌ Ошибка: Запустите скрипт из корневой директории RAGAS"
    exit 1
fi

# Создаем необходимые директории
mkdir -p logs
mkdir -p results
mkdir -p datasets/sberquad/vector_db

# Проверяем, существует ли датасет SberQuAD
if [ ! -f "datasets/sberquad/documents_for_rag.json" ]; then
    echo "📄 Загрузка датасета SberQuAD..."
    python load_sberquad.py
fi

# Проверяем, существует ли векторная БД
if [ ! -d "datasets/sberquad/vector_db" ] || [ ! "$(ls -A datasets/sberquad/vector_db)" ]; then
    echo "🔍 Создание векторной БД..."
    python test_rag.py --config config/base_config.yaml --rebuild-vector-db --max-samples 100
fi

echo "✅ Подготовка завершена"
echo "🌐 Запуск веб-интерфейса..."
echo "📱 Откройте браузер и перейдите по адресу: http://localhost:7860"
echo "🛑 Для остановки нажмите Ctrl+C"
echo ""

# Запускаем веб-интерфейс
python app.py --config config/base_config.yaml --interface gradio
