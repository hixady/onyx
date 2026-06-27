import argparse
import sys

from onyx import RVCModel


def infer(args):
    model = RVCModel(
        model=args.model,
        rmvpe=args.rmvpe,
        cv=args.cv,
        index=args.index,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        verify=not args.no_verify,
    )
    with model:
        result = model.convert(
            input_path=args.input,
            output_path=args.output,
            speaker_id=args.speaker_id,
            f0_method=args.f0_method,
            index_rate=args.index_rate,
        )
        print(f"Saved to {result['output_path']} ({result['sr']} Hz, "
              f"{result['duration']:.1f}s in {result['elapsed']:.1f}s, "
              f"{result['rtf']:.1f}x real-time)")


def pack(args):
    RVCModel.pack(
        output_path=args.output,
        model=args.model,
        index=args.index,
        rmvpe=args.rmvpe,
        cv=args.cv,
        name=args.name,
        author=args.author,
        notes=args.notes,
        tags=args.tags,
        demo=args.demo,
        icon=args.icon,
    )


def unpack(args):
    RVCModel.unpack(args.model, args.output)


def verify(args):
    RVCModel.verify(args.model)


def main():
    parser = argparse.ArgumentParser(description="onyx — RVC voice conversion")
    sub = parser.add_subparsers(dest="command")

    p_infer = sub.add_parser("infer", help="Run voice conversion inference")
    p_infer.add_argument("-i", "--input", required=True)
    p_infer.add_argument("-o", "--output", default="tmp/output.wav")
    p_infer.add_argument("--cv", default=None)
    p_infer.add_argument("--rmvpe", default=None)
    p_infer.add_argument("--model", required=True)
    p_infer.add_argument("--index", default=None)
    p_infer.add_argument("--speaker-id", type=int, default=0)
    p_infer.add_argument("--f0-method", choices=["rmvpe", "autocorr"], default="rmvpe")
    p_infer.add_argument("--index-rate", type=float, default=0.5)
    p_infer.add_argument("--chunk-sec", type=float, default=10)
    p_infer.add_argument("--overlap-sec", type=float, default=0.5)
    p_infer.add_argument("--no-verify", action="store_true", help="Skip sha256 integrity check for .rvc files")
    p_infer.set_defaults(func=infer)

    p_pack = sub.add_parser("pack", help="Package model into .rvc container")
    p_pack.add_argument("-o", "--output", required=True)
    p_pack.add_argument("--model", required=True)
    p_pack.add_argument("--index", default=None)
    p_pack.add_argument("--rmvpe", default=None)
    p_pack.add_argument("--cv", default=None)
    p_pack.add_argument("--name", default=None)
    p_pack.add_argument("--author", default=None)
    p_pack.add_argument("--notes", default=None)
    p_pack.add_argument("--tags", default=None)
    p_pack.add_argument("--demo", default=None)
    p_pack.add_argument("--icon", default=None)
    p_pack.set_defaults(func=pack)

    p_unpack = sub.add_parser("unpack", help="Extract .rvc container to directory")
    p_unpack.add_argument("model")
    p_unpack.add_argument("-o", "--output", required=True)
    p_unpack.set_defaults(func=unpack)

    p_verify = sub.add_parser("verify", help="Verify .rvc container integrity")
    p_verify.add_argument("model")
    p_verify.set_defaults(func=verify)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
