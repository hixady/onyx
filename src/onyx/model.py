import os
from pathlib import Path

from onyx.container import RVCContainer, create_rvc_package
from onyx.pipeline import Pipeline


class RVCModel:
    """RVC voice conversion model.

    Load from a .rvc container or from separate .onnx files:
        model = RVCModel("Cartman.rvc")
        model = RVCModel("Cartman.onnx", rmvpe="rmvpe.onnx", cv="contentvec.onnx")
        model.convert("input.wav", "output.wav")
    """

    def __init__(self, model, rmvpe=None, cv=None, index=None,
                 providers=None, chunk_sec=10, overlap_sec=0.5,
                 verify=True):
        path = Path(model)
        self._container = None
        self._pipeline = None

        if path.suffix.lower() == ".rvc":
            container = RVCContainer(path)
            if verify:
                errors = container.verify()
                if errors:
                    container.close()
                    raise ValueError(f"Corrupted .rvc container: {'; '.join(errors)}")
            meta = container.read_metadata()
            self._meta = meta
            self._name = meta.get("name", path.stem)
            self._container = container

            model_src = container.read_by_role("rvc", "model.onnx")
            rmvpe_src = rmvpe or (container.read_by_role("pitch", "rmvpe.onnx") if container.has_rmvpe else None)
            cv_src = cv or (container.read_by_role("embedding", "contentvec.onnx") if container.has_contentvec else None)
            index_src = index or (container.read_by_role("index", "model.index") if container.has_index else None)
        else:
            self._meta = {}
            self._name = path.stem
            model_src = str(path)
            rmvpe_src = rmvpe
            cv_src = cv
            index_src = index

        if rmvpe_src is None:
            fallback = Path("rmvpe.onnx")
            if fallback.exists():
                rmvpe_src = str(fallback)
        if cv_src is None:
            fallback = Path("contentvec.onnx")
            if fallback.exists():
                cv_src = str(fallback)

        if rmvpe_src is None or cv_src is None:
            missing = [l for l, p in [("RMVPE", rmvpe_src), ("ContentVec", cv_src)] if p is None]
            raise FileNotFoundError(
                f"Missing required models: {', '.join(missing)}. "
                "Provide paths via rmvpe=... and cv=... parameters."
            )

        self._pipeline = Pipeline(
            cv=cv_src, rmvpe=rmvpe_src, model=model_src,
            index=index_src, providers=providers,
            chunk_sec=chunk_sec, overlap_sec=overlap_sec,
        )

    @property
    def metadata(self):
        return dict(self._meta)

    @property
    def version(self):
        return self._pipeline.synth.version

    @property
    def sample_rate(self):
        return self._pipeline.synth.sr_out

    @property
    def name(self):
        return self._name

    def convert(self, input_path, output_path, speaker_id=0,
                f0_method="rmvpe", index_rate=0.5, trim_silence=0):
        return self._pipeline.convert(
            input_path=input_path, output_path=output_path,
            speaker_id=speaker_id, f0_method=f0_method,
            index_rate=index_rate, trim_silence=trim_silence,
        )

    def close(self):
        if self._container:
            self._container.close()
            self._container = None
        self._pipeline = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def pack(output_path, model, index=None, rmvpe=None, cv=None,
             name=None, author=None, notes=None, tags=None,
             demo=None, icon=None, metadata=None):
        metadata = metadata or {}
        if name: metadata["name"] = name
        if author: metadata["author"] = author
        if notes: metadata["notes"] = notes
        if tags: metadata["tags"] = [t.strip() for t in tags.split(",")]

        if isinstance(model, bytes):
            create_rvc_package(
                output_path=output_path, onnx_bytes=model,
                index_path=index, rmvpe_path=rmvpe, contentvec_path=cv,
                metadata=metadata or None, demo_path=demo, icon_path=icon,
            )
        else:
            create_rvc_package(
                output_path=output_path, onnx_path=model,
                index_path=index, rmvpe_path=rmvpe, contentvec_path=cv,
                metadata=metadata or None, demo_path=demo, icon_path=icon,
            )

    @staticmethod
    def verify(path):
        with RVCContainer(path) as c:
            errors = c.verify()
            meta = c.read_metadata()
        if not errors:
            name = meta.get("name", Path(path).stem)
            print(f"Valid: {name}")
            for k, v in meta.items():
                if k == "sha256":
                    if isinstance(v, dict):
                        print(f"  sha256: {len(v)} files verified")
                    else:
                        print(f"  sha256: {v[:16]}...")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"Invalid: {Path(path).name}")
            for e in errors:
                print(f"  ERROR: {e}")
        return errors

    @staticmethod
    def unpack(path, output_dir):
        path = Path(path)
        with RVCContainer(path) as c:
            dst = c.extract_all(output_dir)
        print(f"Extracted {path.name} to {dst}/")
        return dst
