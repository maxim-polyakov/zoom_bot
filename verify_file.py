# verify_file.py
import os
from pathlib import Path

# Укажите ТОЧНЫЙ путь к файлу
file_path = input("Введите ПОЛНЫЙ путь к sdk.dll (например: D:/zoom-sdk-windows-6.7.2.26830/bin/x64/sdk.dll): ").strip()

path = Path(file_path)

print(f"\n🔍 Проверяю путь: {path}")
print(f"Абсолютный путь: {path.absolute()}")
print(f"Существует: {path.exists()}")
print(f"Это файл: {path.is_file()}")
print(f"Размер: {path.stat().st_size if path.exists() else 0} байт")

if path.exists():
    print("\n✅ Файл существует!")

    # Проверяем права
    print(f"Чтение: {os.access(path, os.R_OK)}")
    print(f"Запись: {os.access(path, os.W_OK)}")
    print(f"Исполнение: {os.access(path, os.X_OK)}")

    # Покажем родительскую директорию
    print(f"\n📁 Родительская директория: {path.parent}")
    print(f"Содержимое директории:")
    for item in path.parent.iterdir():
        print(f"  - {item.name}")
else:
    print("\n❌ Файл не найден!")

    # Проверим похожие файлы в той же директории
    parent = Path(file_path).parent
    if parent.exists():
        print(f"\n🔍 Поиск похожих файлов в {parent}:")
        for item in parent.iterdir():
            if item.is_file() and 'dll' in item.name.lower():
                print(f"  • {item.name}")