import hashlib
import json
import zipfile
from pathlib import Path


class Container:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix.lower() != ".onyx":
            raise ValueError(f"Expected .onyx file, got: {path}")
        self._zip = zipfile.ZipFile(self.path, "r")
        self._meta = json.loads(self._zip.read("metadata.json"))
        for key in ("version", "type", "files", "sha256"):
            if key not in self._meta:
                raise ValueError(f"Missing required metadata key: {key}")

    def get_type(self) -> str:
        return self._meta["type"]

    def get_files(self) -> dict[str, str]:
        return dict(self._meta["files"])

    def read_metadata(self) -> dict:
        return dict(self._meta)

    def read_by_role(self, role: str) -> bytes:
        files = self.get_files()
        if role not in files:
            raise KeyError(f"Role '{role}' not in container files: {list(files)}")
        return self._zip.read(files[role])

    def verify(self) -> list[str]:
        errors = []
        hashes = self._meta.get("sha256", {})
        for arcname, expected in hashes.items():
            if arcname not in self._zip.namelist():
                errors.append(f"missing {arcname}")
                continue
            actual = hashlib.sha256(self._zip.read(arcname)).hexdigest()
            if actual != expected:
                errors.append(f"sha256 mismatch for {arcname}")
        files = self.get_files()
        for role, arcname in files.items():
            if arcname not in self._zip.namelist():
                errors.append(f"file '{arcname}' (role={role}) not in archive")
        return errors

    def extract_all(self, dst_dir):
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for name in self._zip.namelist():
            if name == "metadata.json":
                continue
            (dst / name).write_bytes(self._zip.read(name))
        (dst / "metadata.json").write_text(json.dumps(self._meta, indent=2))
        return str(dst)

    def close(self):
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_package(output_path, model_type, metadata=None, file_data=None):
    """Build a .onyx container from arbitrary file data.

    Args:
        output_path: path to write .onyx (suffix enforced)
        model_type: value for metadata 'type' field
        metadata: dict with at least 'files' mapping roles→archive names.
                  Will have 'version' and 'type' set automatically.
        file_data: dict of {archive_name: bytes} for every file in metadata['files'].
    """
    out = Path(output_path)
    if out.suffix.lower() != ".onyx":
        out = out.with_suffix(".onyx")

    if metadata is None:
        metadata = {}
    metadata.setdefault("version", 1)
    metadata.setdefault("type", model_type)

    if "files" not in metadata:
        raise ValueError("metadata must contain 'files' dict mapping roles→archive names")

    files = metadata["files"]
    if file_data is None:
        file_data = {}

    hashes = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for arcname in set(files.values()) | set(file_data.keys()):
            data = file_data.get(arcname)
            if data is None:
                raise ValueError(f"Missing file data for '{arcname}'")
            hashes[arcname] = hashlib.sha256(data).hexdigest()
            zf.writestr(arcname, data)
        metadata["sha256"] = hashes
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    kb = out.stat().st_size / 1024
    print(f"Created {out} ({kb:.0f} KB)")
    return str(out)
