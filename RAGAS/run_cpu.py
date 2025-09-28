#!/usr/bin/env python3
"""
Скрипт для запуска RAG системы на CPU с разными конфигурациями.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Запуск RAG системы на CPU")
    parser.add_argument(
        "--config", 
        choices=["optimized", "test", "base"], 
        default="optimized",
        help="Выберите конфигурацию: optimized (оптимизированная), test (для тестирования), base (базовая)"
    )
    parser.add_argument(
        "--interface", 
        choices=["gradio", "streamlit"], 
        default="gradio",
        help="Выберите веб-интерфейс"
    )
    parser.add_argument(
        "--mode",
        choices=["web", "experiment", "test_dataset"],
        default="web",
        help="Режим запуска: web (веб-интерфейс), experiment (эксперимент), test_dataset (тест на датасете)"
    )
    
    args = parser.parse_args()
    
    # Определяем конфигурационный файл
    config_map = {
        "optimized": "config/cpu_optimized_config.yaml",
        "test": "config/cpu_test_config.yaml", 
        "base": "config/base_config.yaml"
    }
    
    config_file = config_map[args.config]
    
    # Проверяем существование конфигурационного файла
    if not os.path.exists(config_file):
        print(f"❌ Конфигурационный файл {config_file} не найден!")
        sys.exit(1)
    
    print(f"🚀 Запуск RAG системы на CPU")
    print(f"📋 Конфигурация: {args.config} ({config_file})")
    print(f"🌐 Интерфейс: {args.interface}")
    print(f"⚙️  Режим: {args.mode}")
    print("-" * 50)
    
    # Формируем команду
    if args.mode == "web":
        cmd = [
            "python", "app.py",
            "--config", config_file,
            "--interface", args.interface
        ]
    elif args.mode == "experiment":
        cmd = [
            "python", "main.py",
            "--config", config_file
        ]
    elif args.mode == "test_dataset":
        cmd = [
            "python", "test_rag_dataset.py",
            "--config", config_file
        ]
    
    try:
        # Запускаем команду
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при запуске: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Остановка по запросу пользователя")
        sys.exit(0)

if __name__ == "__main__":
    main()






