import argparse
import sys
from pathlib import Path

from onyx import run
from onyx.core.container import Container, create_package
from onyx.core.registry import add_run_subparsers, list_types
from onyx.core.resolver import SHARED_DIR


def cmd_run(args):
    kwargs = {k: v for k, v in vars(args).items()
              if k not in ("func", "command", "arch_type", "input", "output",
                           "model", "chunk_sec", "overlap_sec") and v is not None}
    run(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        arch_type=args.arch_type,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        **kwargs,
    )


def cmd_pack(args):
    files = {}
    file_data = {}
    for role, attr in [("model", "model"), ("pitch", "rmvpe"), ("embedding", "cv"),
                        ("index", "index"), ("demo", "demo"), ("icon", "icon")]:
        path = getattr(args, attr, None)
        if path and Path(path).exists():
            arcname = Path(path).name
            files[role] = arcname
            file_data[arcname] = Path(path).read_bytes()
    meta = {"files": files}
    create_package(output_path=args.output, model_type=args.type,
                   metadata=meta, file_data=file_data)


def cmd_unpack(args):
    with Container(args.model) as c:
        dst = c.extract_all(args.output)
    print(f"Extracted {Path(args.model).name} to {dst}/")


def cmd_verify(args):
    with Container(args.model) as c:
        errors = c.verify()
        meta = c.read_metadata()
    if not errors:
        print(f"Valid: {Path(args.model).stem}")
        for k, v in meta.items():
            if k == "sha256":
                if isinstance(v, dict):
                    print(f"  sha256: {len(v)} files verified")
                else:
                    print(f"  sha256: {str(v)[:16]}...")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"Invalid: {Path(args.model).name}")
        for e in errors:
            print(f"  ERROR: {e}")
    return errors


def cmd_models(args):
    types = list_types()
    print("Registered model types:")
    for t in types:
        print(f"  {t}")
    print(f"\nCache dir: {SHARED_DIR}")


def main():
    parser = argparse.ArgumentParser(description="onyx — ONNX inference framework")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run inference with a model type")
    run_sub = p_run.add_subparsers(dest="arch_type")
    add_run_subparsers(run_sub)
    p_run.set_defaults(func=cmd_run)

    p_pack = sub.add_parser("pack", help="Package files into .onyx container")
    p_pack.add_argument("-o", "--output", required=True)
    p_pack.add_argument("--type", required=True, help="Model type (e.g. rvc)")
    p_pack.add_argument("--model", required=True)
    p_pack.add_argument("--index", default=None)
    p_pack.add_argument("--rmvpe", default=None)
    p_pack.add_argument("--cv", default=None)
    p_pack.add_argument("--demo", default=None)
    p_pack.add_argument("--icon", default=None)
    p_pack.set_defaults(func=cmd_pack)

    p_unpack = sub.add_parser("unpack", help="Extract .onyx container to directory")
    p_unpack.add_argument("model")
    p_unpack.add_argument("-o", "--output", required=True)
    p_unpack.set_defaults(func=cmd_unpack)

    p_verify = sub.add_parser("verify", help="Verify .onyx container integrity")
    p_verify.add_argument("model")
    p_verify.set_defaults(func=cmd_verify)

    p_models = sub.add_parser("models", help="List registered model types and cache")
    p_models.set_defaults(func=cmd_models)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
