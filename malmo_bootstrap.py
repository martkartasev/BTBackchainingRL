import malmo.minecraftbootstrap

from utils.file import get_absolute_path


def bootstrap_malmo():
    malmo.minecraftbootstrap.malmo_install_dir = get_absolute_path("bootstrap/MalmoPlatform")
    malmo.minecraftbootstrap.download()


def run_malmo(ports=[10000, ]):
    malmo.minecraftbootstrap.malmo_install_dir = get_absolute_path("bootstrap/MalmoPlatform")
    malmo.minecraftbootstrap.set_malmo_xsd_path()
    malmo.minecraftbootstrap.launch_minecraft(ports=ports)


# bootstrap_malmo()
run_malmo()
