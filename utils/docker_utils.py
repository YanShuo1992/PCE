import io
import logging
import threading
from pathlib import Path
import tarfile
from typing import Optional, Union
import docker

_thread_local = threading.local()

def get_thread_logger():
    return getattr(_thread_local, 'logger', None)

def setup_logger(log_file):
    thread_id = threading.get_ident()
    logger_name = f'selfimprove_logger_{thread_id}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.stream.lock = threading.Lock()

    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    _thread_local.logger = logger

    return logger

def safe_log(message: str, level: int = logging.INFO):
    logger = get_thread_logger()
    if logger:
        logger.log(level, message)
    else:
        print(f"Warning: No logger found for thread {threading.get_ident()}")

def remove_existing_container(client, container_name):
    try:
        existing_container = client.containers.get(container_name)
        safe_log(f"Removing existing container with name {container_name}")
        existing_container.stop()
        existing_container.remove()
    except docker.errors.NotFound:
        safe_log(f"No existing container with name {container_name} found.")
    except docker.errors.APIError as e:
        safe_log(f"Error removing existing container {container_name}: {e}", logging.ERROR)
        raise

def create_archive(path: Union[str, Path], data: Optional[bytes] = None) -> bytes:
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
        if data is not None:
            tarinfo = tarfile.TarInfo(name=str(path))
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
        else:
            path = Path(path)
            arcname = path.name
            tar.add(path, arcname=arcname)

    tar_stream.seek(0)
    return tar_stream.read()

def build_dgm_container(
        client,
        repo_path='./',
        image_name='app',
        container_name='app-container',
        force_rebuild=False,
    ):
    try:
        existing = [
            r for r in client.api.images()
            if any(t == image_name or t.rsplit(":", 1)[0] == image_name
                   for t in (r.get("RepoTags") or []))
        ]
        if force_rebuild or not existing:
            safe_log("Building the Docker image...")
            image, logs = client.images.build(path=repo_path, tag=image_name, rm=True)
            for log_entry in logs:
                if 'stream' in log_entry:
                    safe_log(log_entry['stream'].strip())
            safe_log("Image built successfully.")
        else:
            safe_log(f"Docker image '{image_name}' already exists. Skipping build.")
            image = existing[0]
    except Exception as e:
        safe_log(f"Error while building the Docker image: {e}")
        raise RuntimeError(f"build_dgm_container: could not build image {image_name!r}: {e}") from e

    try:
        container = client.containers.run(image=image_name, name=container_name, detach=True)
        safe_log(f"Container '{container_name}' started successfully.")
        return container
    except Exception as e:
        safe_log(f"Error while starting the container: {e}")
        raise RuntimeError(f"build_dgm_container: could not start container {container_name!r}: {e}") from e

def cleanup_container(container):
    safe_log(f"Stopping container '{container.name}'...")
    container.stop()
    container.remove()
    safe_log(f"Container '{container.name}' removed.")

def copy_to_container(container, source_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        if source_path.is_file():
            container_dest_dir = str(dest_path.parent)
            archive_path = dest_path.name
            with open(source_path, 'rb') as source_file:
                data = source_file.read()
            archive = create_archive(archive_path, data)
        else:
            container_dest_dir = str(dest_path.parent)
            archive = create_archive(source_path)

        container.exec_run(f"mkdir -p {container_dest_dir}")

        safe_log(f"Copying {source_path} to container at {dest_path}")
        success = container.put_archive(container_dest_dir, archive)

        if not success:
            raise Exception(f"Failed to copy {source_path} to container")

        safe_log(f"Successfully copied {source_path} to container")

    except Exception as e:
        safe_log(f"Error copying to container: {e}", logging.ERROR)
        raise

def copy_from_container(container, source_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        result = container.exec_run(f"test -e {source_path}")
        if result.exit_code and result.exit_code != 0:
            raise FileNotFoundError(f"Source path not found in container: {source_path}")

        result = container.exec_run(f"stat -f '%HT' {source_path}")
        is_file = result.output.decode().strip() == 'Regular File'

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        safe_log(f"Copying from container {source_path} to local path {dest_path}")

        bits, stat = container.get_archive(str(source_path))

        archive_data = b''.join(bits)

        stream = io.BytesIO(archive_data)

        with tarfile.open(fileobj=stream, mode='r') as tar:
            if is_file:
                member = tar.getmembers()[0]
                with tar.extractfile(member) as source_file:
                    data = source_file.read()
                    with open(dest_path, 'wb') as dest_file:
                        dest_file.write(data)
            else:
                tar.extractall(path=str(dest_path.parent))
                extracted_path = dest_path.parent / Path(stat['name']).name
                if extracted_path != dest_path and extracted_path.exists():
                    extracted_path.rename(dest_path)

        safe_log(f"Successfully copied from container to {dest_path}")

    except Exception as e:
        safe_log(f"Error copying from container: {e}", logging.ERROR)
        raise

def log_container_output(exec_result):
    if isinstance(exec_result.output, bytes):
        safe_log(f"Container output: {exec_result.output.decode()}")
    else:
        for chunk in exec_result.output:
            if chunk:
                safe_log(f"Container output: {chunk.decode().strip()}")

    if exec_result.exit_code and exec_result.exit_code != 0:
        error_msg = f"Script failed with exit code {exec_result.exit_code}"
        safe_log(error_msg, logging.ERROR)
        raise Exception(error_msg)
