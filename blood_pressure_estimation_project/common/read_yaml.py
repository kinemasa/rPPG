# common/config_loader.py

import yaml
from pathlib import Path

def load_config(config_path=None):
    """YAML設定ファイルを読み込んで辞書として返す"""
    if config_path is None:
        config_path = Path(__file__).parent / "../config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config