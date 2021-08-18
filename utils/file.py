from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def get_absolute_path(relative_path):
    return str(get_project_root() / relative_path)
