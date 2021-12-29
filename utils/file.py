import codecs
import os
from pathlib import Path
from shutil import copyfile

import jsonpickle
import simplejson as simplejson


def get_project_root():
    return Path(__file__).parent.parent


def get_absolute_path(relative_path):
    return str(get_project_root() / relative_path)


def create_file_and_write(file_name, function):
    file_path = get_project_root() / file_name
    folder_path = file_path.parent
    folder_path.mkdir(parents=True, exist_ok=True)
    with codecs.open(str(file_path), "w+", "utf-8") as file:
        function(file)


def store_spec(spec):
    spec_pkl = str(Path(spec["model_log_dir"]) / "spec.pkl")
    mission_xml = str(Path(spec["model_log_dir"]) / "mission.xml")
    create_file_and_write(spec_pkl, lambda file: file.write(
        simplejson.dumps(simplejson.loads(jsonpickle.encode(spec)), indent=4)))
    copyfile(get_absolute_path(spec['mission']), mission_xml)


def load_spec(model_log_dir):
    with open(get_absolute_path(model_log_dir) + "/spec.pkl", 'rb') as f:
        spec = f.read()
    decode = jsonpickle.decode(spec)
    decode['mission'] = decode['model_log_dir'] + "/mission.xml"
    return decode


def get_model_file_names_from_folder(log_dir):
    result_files = os.listdir(get_absolute_path(log_dir))
    model_files = [result_file.rstrip('.zip') for result_file in result_files if result_file.endswith('.zip')]
    return model_files
