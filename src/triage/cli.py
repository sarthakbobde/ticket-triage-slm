"""The `triage` command: one entry point for every step of the pipeline.

    triage data download          fetch and profile the raw dataset         (Step 02)
    triage data splits            build the test / val / train splits       (Step 03)
    triage data mine --round 2    find extra rare-class tickets             (Step 03)
    triage data merge-hr          apply ADR-013 to the teacher labels       (Step 03)
    triage data gold-sheet        build the hand-labelling page             (Step 04)
    triage data training-data     build the Kaggle training files           (Step 07)
    triage label --split train    label tickets with the teacher            (Step 03)
    triage pilot-report           compare teacher candidates                (Step 03)
    triage eval run ...           collect a model's answers on the gold set (Step 05)
    triage eval score --run NAME  score those answers                       (Step 05)
    triage eval calibrate         choose the gateway confidence threshold   (Step 11)
    triage serve                  run the HTTP gateway                      (Step 11)
    triage export merge | ollama  merge the adapter / import into Ollama    (Step 09)

How this file is organised:
  1. One small function per command (the "handlers"). Each one only reads the options the person
     typed and calls a function that does the real work, so that work can be tested without a terminal.
  2. build_parser() lists which words and options exist, and which handler each command runs.
  3. main() reads what was typed and runs the right handler.

The handlers import their code INSIDE the function on purpose: it keeps `triage --help` instant,
because some of those packages (pandas, datasets, torch) are slow to load.
"""

import argparse
import logging
import sys
from pathlib import Path

from triage import paths
from triage.errors import UserError

log = logging.getLogger("triage")


# ---- 1. the handlers: one small function per command ----------------------------------------


def cmd_data_download(args):
    from triage.data.download import download_and_profile

    download_and_profile()


def cmd_data_splits(args):
    from triage.data.splits import build_splits

    build_splits(force=args.force)


def cmd_data_mine(args):
    from triage.data.mining import mine_rare_classes

    mine_rare_classes(args.round)


def cmd_data_merge_hr(args):
    from triage.data.merge import merge_hr_labels

    merge_hr_labels(teacher=args.teacher)


def cmd_data_gold_sheet(args):
    from triage.data.gold_sheet import build_gold_sheet

    build_gold_sheet()


def cmd_data_training_data(args):
    from triage.data.training import build_training_files

    build_training_files()


def cmd_label(args):
    from triage.labeling.teacher import label_split, list_models
    from triage.llm import make_client
    from triage.settings import load_endpoint

    if args.list_models:
        client = make_client(load_endpoint("teacher"))
        for model_id in list_models(client):
            print(model_id)
        return
    if not args.split:
        raise UserError("--split is required (or use --list-models).")
    label_split(args.split, model=args.model, limit=args.limit, approved=args.yes)


def cmd_pilot_report(args):
    from triage.labeling.pilot import build_pilot_report

    build_pilot_report()


def cmd_eval_run(args):
    from triage.evals.runner import run_model

    run_model(args.config, args.run, model=args.model, fmt=args.format, limit=args.limit, approved=args.yes)


def cmd_eval_score(args):
    from triage.evals.scoring import score_run

    summary, lines = score_run(args.run)
    print("\n".join(lines))


def cmd_eval_calibrate(args):
    from triage.gateway.calibrate import analyse, collect_student_answers

    if not args.analyse_only:
        collect_student_answers(args.student, args.run)
    lines = analyse(args.run, args.teacher_run)
    print("\n".join(lines))


def cmd_export_merge(args):
    from triage.export.merge import merge_adapter

    merge_adapter(Path(args.adapter), Path(args.out))


def cmd_export_ollama(args):
    from triage.export.ollama import create_ollama_model

    create_ollama_model(Path(args.gguf), args.name)


def cmd_serve(args):
    import uvicorn

    from triage.gateway.app import create_app
    from triage.gateway.service import build_service

    service = build_service(
        student_config=args.student,
        use_teacher=args.teacher,
        approved=args.yes,
        teacher_budget=args.teacher_budget,
        min_confidence=args.min_confidence,
    )
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


# ---- 2. the parser: which words and options exist -------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(prog="triage", description="Ticket-triage pipeline.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    commands = parser.add_subparsers(dest="command", required=True)

    # triage data ...
    data_parser = commands.add_parser("data", help="build and prepare datasets")
    data = data_parser.add_subparsers(dest="action", required=True)

    download = data.add_parser("download", help="download and profile the raw dataset")
    download.set_defaults(handler=cmd_data_download)

    splits = data.add_parser("splits", help="build test / val / train splits")
    splits.add_argument("--force", action="store_true", help="overwrite the frozen test set (do not)")
    splits.set_defaults(handler=cmd_data_splits)

    mine = data.add_parser("mine", help="find extra rare-class tickets")
    mine.add_argument("--round", type=int, required=True, choices=[1, 2])
    mine.set_defaults(handler=cmd_data_mine)

    merge = data.add_parser("merge-hr", help="merge hr into customer_service in the teacher labels")
    merge.add_argument("--teacher", default="gpt-4.1")
    merge.set_defaults(handler=cmd_data_merge_hr)

    gold_sheet = data.add_parser("gold-sheet", help="build the hand-labelling page")
    gold_sheet.set_defaults(handler=cmd_data_gold_sheet)

    training_data = data.add_parser("training-data", help="build the Kaggle training files")
    training_data.set_defaults(handler=cmd_data_training_data)

    # triage label ...
    label = commands.add_parser("label", help="label tickets with the teacher model (costs money)")
    label.add_argument("--split", choices=["val", "train", "train_extra", "train_extra2", "test"])
    label.add_argument("--model", help="override teacher.model in configs/models.yaml")
    label.add_argument("--limit", type=int, help="only the first N tickets of the split")
    label.add_argument("--list-models", action="store_true", help="print the model ids the key can use")
    label.add_argument("--yes", action="store_true", help="allow paid calls to a remote endpoint")
    label.set_defaults(handler=cmd_label)

    pilot = commands.add_parser("pilot-report", help="compare teacher candidates")
    pilot.set_defaults(handler=cmd_pilot_report)

    # triage eval ...
    eval_parser = commands.add_parser("eval", help="run and score models on the gold set")
    evaluate = eval_parser.add_subparsers(dest="action", required=True)

    run = evaluate.add_parser("run", help="collect a model's raw answers on the gold tickets")
    run.add_argument("--config", required=True, help="section of configs/models.yaml, e.g. base_qwen")
    run.add_argument("--run", required=True, help="name for this run, e.g. base-qwen-none")
    run.add_argument("--model", help="override the model name in the config")
    run.add_argument("--format", choices=["none", "json_object", "json_schema"], default="none")
    run.add_argument("--limit", type=int, help="only the first N gold tickets")
    run.add_argument("--yes", action="store_true", help="allow paid calls to a remote endpoint")
    run.set_defaults(handler=cmd_eval_run)

    score = evaluate.add_parser("score", help="score a run against the gold labels (free)")
    score.add_argument("--run", required=True)
    score.set_defaults(handler=cmd_eval_score)

    # triage serve
    serve = commands.add_parser("serve", help="run the HTTP gateway (student first, teacher as fallback)")
    serve.add_argument("--student", default="student_q4", help="section of configs/models.yaml for the small model")
    serve.add_argument(
        "--min-confidence", type=float, default=0.0, help="answers less sure than this go to the teacher"
    )
    serve.add_argument("--teacher", action="store_true", help="allow falling back to the paid teacher model")
    serve.add_argument("--teacher-budget", type=int, default=0, help="the most paid teacher calls allowed this run")
    serve.add_argument("--yes", action="store_true", help="confirm that paid teacher calls are allowed")
    serve.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this machine only")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=cmd_serve)

    # triage export ...
    export_parser = commands.add_parser("export", help="merge the adapter and import into Ollama")
    export = export_parser.add_subparsers(dest="action", required=True)

    merge_cmd = export.add_parser("merge", help="merge the LoRA adapter into the base model (needs the export extra)")
    merge_cmd.add_argument("--adapter", default=str(paths.ADAPTERS / "qwen-triage-lora-v1"))
    merge_cmd.add_argument("--out", default=str(paths.MERGED_MODELS / "qwen-triage-v1"))
    merge_cmd.set_defaults(handler=cmd_export_merge)

    ollama_cmd = export.add_parser("ollama", help="import a GGUF file into Ollama")
    ollama_cmd.add_argument("--gguf", required=True, help="path to the .gguf file, e.g. models/gguf/...-Q4_K_M.gguf")
    ollama_cmd.add_argument("--name", required=True, help="Ollama model name, e.g. ticket-triage-q4")
    ollama_cmd.set_defaults(handler=cmd_export_ollama)

    # triage eval calibrate
    calibrate = evaluate.add_parser("calibrate", help="choose the gateway confidence threshold (student runs locally)")
    calibrate.add_argument("--student", default="student_q4", help="section of configs/models.yaml")
    calibrate.add_argument("--run", default="gateway-student-q4", help="name for the student answers file")
    calibrate.add_argument("--teacher-run", default="teacher-gpt-4.1", help="an existing teacher run to reuse (free)")
    calibrate.add_argument("--analyse-only", action="store_true", help="skip asking the student, use the saved file")
    calibrate.set_defaults(handler=cmd_eval_calibrate)

    return parser


# ---- 3. main: read what was typed and run the right handler ---------------------------------


def main(argv=None):
    # Windows consoles default to an old text encoding that crashes on some characters in tickets.
    sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(message)s")

    # args.handler is the function that build_parser() attached to the chosen command.
    handler = args.handler
    try:
        handler(args)
    except UserError as error:
        log.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
