import argparse

import numpy as np


class ModelArch:
    type: str = ""
    description: str = ""
    shared_models: dict[str, list[str]] = {}
    chunkable: bool = True
    supports: list[str] = ["audio"]

    @staticmethod
    def add_args(p: argparse.ArgumentParser):
        pass

    def validate(self, container):
        pass

    def run(self, audio: np.ndarray, sr: int, container,
            shared: dict[str, bytes], **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError
