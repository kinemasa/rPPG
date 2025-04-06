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