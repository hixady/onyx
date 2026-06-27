import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


class RVCContainer:
    def __init__(self, path):
        self.path = Path(path)
        self._zip = None
        self._tmpdir = None
        self._extracted = {}

    def _open(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.path, "r")
        return self._zip

    def _names(self):
        return set(self._open().namelist())

    def _ensure_tmpdir(self):
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="onyx_")
        return self._tmpdir

    @property
    def has_index(self):
        return "model.index" in self._names()

    @property
    def has_rmvpe(self):
        return "rmvpe.onnx" in self._names()

    @property
    def has_contentvec(self):
        return "contentvec.onnx" in self._names()

    @property
    def has_metadata(self):
        return "metadata.json" in self._names()

    @property
    def has_demo(self):
        return "demo.wav" in self._names()

    @property
    def has_icon(self):
        return "icon.png" in self._names()

    def read(self, name):
        return self._open().read(name)

    def read_metadata(self):
        if self.has_metadata:
            return json.loads(self.read("metadata.json"))
        return {}

    def extract(self, name):
        if name in self._extracted:
            return self._extracted[name]
        dst = os.path.join(self._ensure_tmpdir(), name)
        with open(dst, "wb") as f:
            f.write(self.read(name))
        self._extracted[name] = dst
        return dst

    def extract_all(self, dst_dir):
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for name in self._names():
            if name == "metadata.json":
                continue
            (dst / name).write_bytes(self.read(name))
        meta = self.read_metadata()
        if meta:
            (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
        return str(dst)

    def verify(self):
        errors = []

        if not self.has_metadata:
            errors.append("missing metadata.json")
            return errors

        meta = self.read_metadata()
        sha256_expected = meta.get("sha256")

        if sha256_expected:
            data = self.read("model.onnx")
            sha256_actual = hashlib.sha256(data).hexdigest()
            if sha256_actual != sha256_expected:
                errors.append(
                    f"sha256 mismatch: expected {sha256_expected}, got {sha256_actual}"
                )

        required = ["model.onnx"]
        for name in required:
            if name not in self._names():
                errors.append(f"missing {name}")

        return errors

    def close(self):
        if self._zip:
            self._zip.close()
            self._zip = None
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_rvc_package(output_path, onnx_path=None, onnx_bytes=None,
                       index_path=None, rmvpe_path=None, contentvec_path=None,
                       metadata=None, demo_path=None, icon_path=None):
    out = Path(output_path)
    if out.suffix.lower() != ".rvc":
        out = out.with_suffix(".rvc")

    if metadata is None:
        metadata = {}

    if onnx_bytes is None:
        onnx_bytes = Path(onnx_path).read_bytes()

    sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    metadata["sha256"] = sha256

    if "sample_rate" not in metadata or "phone_dim" not in metadata:
        import onnx
        model = onnx.load_model_from_string(onnx_bytes)

    if "sample_rate" not in metadata:
        ratio = 1
        for node in model.graph.node:
            if node.op_type == "ConvTranspose":
                for attr in node.attribute:
                    if attr.name == "strides":
                        ratio *= attr.ints[0]
        metadata["sample_rate"] = ratio * 100

    if "phone_dim" not in metadata:
        phone_dim = 768
        for inp in model.graph.input:
            if inp.name == "phone":
                dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                if len(dims) == 3 and dims[2] > 0:
                    phone_dim = dims[2]
        metadata["phone_dim"] = phone_dim

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.onnx", onnx_bytes)
        for opt_path, arcname in [(index_path, "model.index"),
                                  (rmvpe_path, "rmvpe.onnx"),
                                  (contentvec_path, "contentvec.onnx"),
                                  (demo_path, "demo.wav"),
                                  (icon_path, "icon.png")]:
            if opt_path and Path(opt_path).exists():
                zf.write(opt_path, arcname)
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    kb = out.stat().st_size / 1024
    print(f"Created {out} ({kb:.0f} KB)")
    return str(out)
