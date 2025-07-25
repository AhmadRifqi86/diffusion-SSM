
import subprocess
from pathlib import Path


def run_cmd(cmd, capture_output=False):
    """Run a shell command and return output or status."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
    if capture_output:
        return result.stdout.strip()
    return result.returncode == 0

def docker_exec(name, exec_cmd):
    print(f"Running command inside container: {name}")
    cmd = f"docker exec {name} {exec_cmd}"
    run_cmd(cmd)

def container_exists(name):
    cmd = f"docker ps -a --format '{{{{.Names}}}}' | grep -w {name}"
    return run_cmd(cmd)

def container_running(name):
    cmd = f"docker ps --format '{{{{.Names}}}}' | grep -w {name}"
    return run_cmd(cmd)

def start_container(name):
    print(f"Starting existing container: {name}")
    run_cmd(f"docker start {name}")

def run_new_container(name, image): #image name either tensorrt or triton
    print(f"Running new TensorRT container: {name}")
    # Get the project root directory (parent of deploy/)
    project_root = Path(__file__).parent.parent.resolve()
    print(f"📁 Project root: {project_root}")
    print(f"📁 Mounting to container: /workspace")
    run_cmd(
        f"docker run --gpus=all -dit --name {name} -v {project_root}:/workspace {image} bash"
    )