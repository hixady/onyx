from pathlib import Path

from onyx.core.arch import ModelArch
from onyx.core.container import Container

SHARED_DIR = Path.home() / ".cache" / "onyx" / "shared"


def resolve_shared(arch: ModelArch, container: Container,
                   cli_overrides: dict) -> dict[str, bytes]:
    result = {}
    for role, candidates in arch.shared_models.items():
        src = cli_overrides.get(role)
        if src:
            result[role] = Path(src).read_bytes()
            continue
        try:
            result[role] = container.read_by_role(role)
            continue
        except KeyError:
            pass
        for name in candidates:
            cache_path = SHARED_DIR / name
            if cache_path.exists():
                result[role] = cache_path.read_bytes()
                break
        else:
            raise FileNotFoundError(
                f"Missing required model '{role}'. "
                f"Install: onyx models install {arch.type} "
                f"or specify --{role} <path>"
            )
    return result
