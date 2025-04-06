# color_vector/utils.py

import tkinter as tk
from tkinter import filedialog

def select_folder(message="フォルダを選択してください"):
    root = tk.Tk()
    root.withdraw()
    print(message)
    folder_path = filedialog.askdirectory(title=message)
    if folder_path:
        print("選択されたフォルダ:", folder_path)
    else:
        print("フォルダが選択されませんでした。")
    return folder_path

def select_file(message="ファイルを選択してください", filetypes=(("画像ファイル", "*.png;*.jpg;*.jpeg"), ("すべてのファイル", "*.*"))):
    root = tk.Tk()
    root.withdraw()
    print(message)
    file_path = filedialog.askopenfilename(title=message, filetypes=filetypes)
    if file_path:
        print("選択されたファイル:", file_path)
    else:
        print("ファイルが選択されませんでした。")
    return file_path