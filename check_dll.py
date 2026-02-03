# check_dll.py
import os
import ctypes
from pathlib import Path
import sys


def inspect_dll(dll_path):
    """Исследовать DLL файл"""
    print(f"🔍 Исследую DLL: {dll_path}")

    if not Path(dll_path).exists():
        print(f"❌ Файл не найден: {dll_path}")
        return

    try:
        # Пробуем загрузить DLL
        dll = ctypes.WinDLL(str(dll_path))
        print(f"✅ DLL успешно загружена")

        # Получаем список экспортируемых функций
        print("\n📋 Поиск функций...")

        # Попробуем найти стандартные функции
        common_functions = [
            'Initialize',
            'InitSDK',
            'ZoomSDK_Initialize',
            'CreateMeeting',
            'JoinMeeting',
            'GetSDKVersion',
            'GetVersion',
            'Cleanup',
            'Destroy',
            'StartMeeting',
            'Login',
            'Logout'
        ]

        found_functions = []
        for func_name in common_functions:
            try:
                func = getattr(dll, func_name, None)
                if func:
                    found_functions.append(func_name)
                    print(f"  ✓ Найдена функция: {func_name}")
            except:
                pass

        print(f"\n🎯 Найдено функций: {len(found_functions)}")

        if found_functions:
            print("\n🔧 Доступные функции:")
            for func in found_functions[:10]:  # Покажем первые 10
                print(f"  - {func}")

        # Попробуем вызвать GetVersion если есть
        if 'GetVersion' in found_functions:
            try:
                dll.GetVersion.restype = ctypes.c_char_p
                version = dll.GetVersion()
                if version:
                    print(f"\n📦 Версия SDK: {version.decode()}")
            except:
                pass

        if 'ZoomSDK_GetVersion' in found_functions:
            try:
                dll.ZoomSDK_GetVersion.restype = ctypes.c_char_p
                version = dll.ZoomSDK_GetVersion()
                if version:
                    print(f"\n📦 Версия Zoom SDK: {version.decode()}")
            except:
                pass

    except Exception as e:
        print(f"❌ Ошибка при загрузке DLL: {e}")


if __name__ == "__main__":
    # Пути для проверки
    check_paths = []

    # Добавьте свой путь здесь!
    custom_path = input("Введите путь к sdk.dll (или нажмите Enter для поиска): ").strip()
    if custom_path:
        check_paths.append(custom_path)


    # Автопоиск
    check_paths.extend([
        "sdk.dll",
        "bin/x64/sdk.dll",
        "bin/x86/sdk.dll",
        "zoom-sdk-windows-6.7.2.26830/bin/x64/sdk.dll",
        "zoom-sdk-windows-6.7.2.26830/bin/x86/sdk.dll",
        "zoom_sdk/bin/x64/sdk.dll",
        "zoom_sdk/bin/x86/sdk.dll",
        "C:/zoom-sdk-windows-6.7.2.26830/bin/x64/sdk.dll",
        "D:/zoom-sdk-windows-6.7.2.26830/bin/x64/sdk.dll",
    ])

    for path in check_paths:
        if Path(path).exists():
            inspect_dll(path)
            break
    else:
        print("❌ sdk.dll не найден. Создайте файл в текущей директории.")