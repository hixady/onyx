import argparse

from onyx.core.arch import ModelArch

_REGISTRY: dict[str, type[ModelArch]] = {}


def register(type_name: str):
    def wrapper(cls):
        _REGISTRY[type_name] = cls
        return cls
    return wrapper


def lookup(type_name: str) -> type[ModelArch]:
    if type_name not in _REGISTRY:
        raise ValueError(f"Unknown model type: {type_name}. Available: {list_types()}")
    return _REGISTRY[type_name]


def list_types() -> list[str]:
    return list(_REGISTRY)


def scan_entry_points():
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="onyx.arch")
        for ep in eps:
            if ep.name not in _REGISTRY:
                _REGISTRY[ep.name] = ep.load()
    except Exception:
        pass


def add_run_subparsers(subparsers: argparse._SubParsersAction):
    for type_name, cls in _REGISTRY.items():
        p = subparsers.add_parser(type_name, help=cls.description)
        p.set_defaults(arch_type=type_name)
        p.add_argument("-i", "--input", required=True)
        p.add_argument("-o", "--output", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--chunk-sec", type=float, default=0)
        p.add_argument("--overlap-sec", type=float, default=0.5)
        cls.add_args(p)
