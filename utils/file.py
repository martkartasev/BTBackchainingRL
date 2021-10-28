import codecs
import json
import pickle
from pathlib import Path
from shutil import copyfile

import jsonpickle


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


def store_spec(spec):
    copyfile(get_absolute_path(spec['mission']), get_absolute_path(spec['model_log_dir']) + "/mission.xml")
    create_file_and_write(spec['model_log_dir'] + "/spec.pkl", lambda file: file.write(jsonpickle.encode(spec)))


def load_spec(model_log_dir):
    with open(get_absolute_path(model_log_dir) + "/spec.pkl", 'rb') as f:
        spec = f.read()
    decode = jsonpickle.decode(spec)
    decode['mission'] = decode['model_log_dir'] + "/mission.xml"
    return decode
