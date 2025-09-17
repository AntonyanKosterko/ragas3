#!/bin/bash

# Скрипт для быстрого запуска RAG экспериментов
# Использование: ./run_experiment.sh [config] [options]

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка аргументов
CONFIG=${1:-"config/base_config.yaml"}
OPTIONS=${@:2}

print_message "Запуск RAG эксперимента с конфигурацией: $CONFIG"

# Проверка существования конфигурации
if [ ! -f "$CONFIG" ]; then
    print_error "Конфигурационный файл не найден: $CONFIG"
    exit 1
fi

# Создание необходимых директорий
print_message "Создание необходимых директорий..."
mkdir -p logs results data/documents notebooks

# Проверка виртуального окружения
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Виртуальное окружение не активировано"
    print_message "Рекомендуется активировать виртуальное окружение:"
    print_message "source rag_env/bin/activate"
fi

# Проверка зависимостей
print_message "Проверка зависимостей..."
python -c "import torch, transformers, langchain, mlflow" 2>/dev/null || {
    print_error "Не все зависимости установлены. Установите их командой:"
    print_error "pip install -r requirements.txt"
    exit 1
}

# Запуск эксперимента
print_message "Запуск эксперимента..."
python main.py --config "$CONFIG" $OPTIONS

if [ $? -eq 0 ]; then
    print_success "Эксперимент завершен успешно!"
    print_message "Результаты сохранены в папке results/"
    print_message "Для просмотра MLflow UI выполните: mlflow ui"
    print_message "Для запуска веб-интерфейса выполните: python app.py --config $CONFIG"
else
    print_error "Эксперимент завершился с ошибкой"
    exit 1
fi

