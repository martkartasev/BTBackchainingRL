import codecs
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def get_absolute_path(relative_path):
    return str(get_project_root() / relative_path)


def create_file_and_write(file_name, function):
    file_path = get_project_root() / file_name
    folder_path = file_path.parent
    folder_path.mkdir(parents=True, exist_ok=True)
    with codecs.open(str(file_path), "w", "utf-8") as file:
        function(file)
