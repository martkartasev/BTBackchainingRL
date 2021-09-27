import malmo.minecraftbootstrap

from utils.file import get_absolute_path


def bootstrap_malmo():
    malmo.minecraftbootstrap.malmo_install_dir = get_absolute_path("bootstrap/MalmoPlatform")
    malmo.minecraftbootstrap.download()


def run_malmo():
    malmo.minecraftbootstrap.malmo_install_dir = get_absolute_path("bootstrap/MalmoPlatform")
    malmo.minecraftbootstrap.set_malmo_xsd_path()
    malmo.minecraftbootstrap.launch_minecraft()


run_malmo()
