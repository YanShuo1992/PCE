import json
import os

def ensure_patch_file_newline(patch_path):
    if not os.path.isfile(patch_path):
        return
    with open(patch_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    if not content.strip():
        return
    lines = content.splitlines()
    normalized = '\n'.join(lines) + '\n'
    if normalized != content:
        with open(patch_path, 'w', encoding='utf-8') as f:
            f.write(normalized)

def read_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    return content

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
