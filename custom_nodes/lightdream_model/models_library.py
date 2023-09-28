import base64
import json
import logging
import os
from pathlib import Path
from typing import List

from google.cloud.storage import Blob, Client
from tqdm import tqdm

_DEFAULT_MODELS_BUCKET = "ltx-research-txt2img-sdk"
_DEFAULT_LOCAL_MODELS_DIR = Path.home() / ".models"

_MODELS_BUCKET = os.environ.get("MODELS_BUCKET", _DEFAULT_MODELS_BUCKET)
_LOCAL_MODELS_DIR = os.environ.get("MODELS_PATH", _DEFAULT_LOCAL_MODELS_DIR)
_LOG = logging.getLogger(__name__)


def fetch(name: str) -> Path:
    """
    Fetches a file/folder from the remote models library, if it does not exist locally.
    :param name: The name of the file/folder to fetch.
    :return: A local path to the fetched file/folder.
    """

    blobs = _list_blobs(name)
    if not blobs:
        raise FileNotFoundError(
            f"File/folder '{name}' does not exist in the models library bucket ({_MODELS_BUCKET})",
        )
    _fetch_blobs(blobs)
    return Path(_LOCAL_MODELS_DIR) / name


def _list_blobs(name: str) -> List[Blob]:
    client = Client()
    bucket = client.get_bucket(_MODELS_BUCKET)
    if Blob(name, bucket).exists():
        blobs = [client.get_bucket(_MODELS_BUCKET).get_blob(name)]
    else:
        name = name.rstrip("/") + "/"
        blobs = [blob for blob in client.list_blobs(bucket, prefix=name) if blob.name != name]

    return blobs


def _fetch_blobs(blobs: List[Blob]) -> None:
    # Load checksums from checksums file
    checksums_path = Path(_LOCAL_MODELS_DIR) / "_checksums.json"
    if checksums_path.exists():
        with open(checksums_path, "r") as f:
            checksums = json.load(f)
    else:
        checksums = {}

    storage_client = Client()
    for i, blob in enumerate(blobs):
        blob_md5_hash = base64.b64decode(blob.md5_hash).hex() if blob.md5_hash else None

        blob_local_file = Path(_LOCAL_MODELS_DIR) / blob.name
        is_local_file_missing = not Path(blob_local_file).exists()
        is_checksum_different = checksums.get(blob.name) != blob_md5_hash

        if is_local_file_missing or is_checksum_different:
            Path(blob_local_file).parent.mkdir(parents=True, exist_ok=True)
            with open(blob_local_file, "wb") as f:
                description = f"{i + 1}/{len(blobs)}: {blob.name}"
                with tqdm.wrapattr(f, "write", total=blob.size, desc=description) as file:
                    storage_client.download_blob_to_file(blob, file)

            # Update checksums
            checksums[blob.name] = blob_md5_hash
            with open(checksums_path, "w") as f:
                json.dump(checksums, f, indent=4)
