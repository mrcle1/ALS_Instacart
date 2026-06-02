"""Top-level convenience CLI.

This script is a thin wrapper so an operator can run the most common
tasks without invoking ``python -m`` on a specific module.

Usage examples::

    python run.py train            # run the full training pipeline
    python run.py infer --algo als
    python run.py serve            # launch the FastAPI service
    python run.py health           # liveness probe (CLI variant)

The script is intentionally minimal: it only routes subcommands to the
already-tested modules. All heavy lifting lives under :mod:`src` and
:mod:`api`.
"""
import argparse
import sys
from pathlib import Path

# Make the project importable regardless of where the script is run from.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import setup_logging
from src.config import get_config


def _cmd_train(_args: argparse.Namespace) -> int:
    from src.pipeline.train_pipeline import run_training_pipeline
    run_training_pipeline()
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    from src.pipeline.infer_pipeline import run_inference_pipeline
    run_inference_pipeline(algorithm=args.algo, N=args.N)
    return 0


def _cmd_serve(_args: argparse.Namespace) -> int:
    import uvicorn

    cfg = get_config()
    uvicorn.run("api.main:app", host=cfg.api.host, port=cfg.api.port, reload=False)
    return 0


def _cmd_health(_args: argparse.Namespace) -> int:
    cfg = get_config()
    from pathlib import Path
    als = Path(cfg.paths.models_dir) / "als_model.pkl"
    fpg = Path(cfg.paths.models_dir) / "fpgrowth_model.pkl"
    enc = Path(cfg.paths.models_dir) / "encoders.pkl"
    ok = als.exists() and fpg.exists() and enc.exists()
    print("OK" if ok else "MODELS-MISSING")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Instacart Recommender CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Run the training pipeline").set_defaults(
        func=_cmd_train)
    p_infer = sub.add_parser("infer", help="Run inference for a cohort")
    p_infer.add_argument("--algo", choices=["als", "fpgrowth"], default="als")
    p_infer.add_argument("--N", type=int, default=10)
    p_infer.set_defaults(func=_cmd_infer)
    sub.add_parser("serve", help="Launch the FastAPI service").set_defaults(
        func=_cmd_serve)
    sub.add_parser("health", help="Check artefact presence").set_defaults(
        func=_cmd_health)

    args = parser.parse_args(argv)
    setup_logging(get_config().paths.logs_dir)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())