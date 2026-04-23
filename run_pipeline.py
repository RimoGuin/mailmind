"""
run_pipeline.py
Orchestrates all 3 stages. Supports:
  - Full run:       python run_pipeline.py --input data/emails.csv
  - Single stage:   python run_pipeline.py --stage load --input data/emails.csv
                    python run_pipeline.py --stage transform
                    python run_pipeline.py --stage enrich
  - With dataset:   python run_pipeline.py --input data/enron.csv --dataset enron
  - Limit (test):   python run_pipeline.py --stage enrich --limit 50
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from db.connection import init_db, get_engine
from pipeline import stage1_load, stage2_transform, stage3_enrich

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ]
)
logger = logging.getLogger("orchestrator")


def stage_load(engine, args):
    if not args.input:
        logger.error("--input required for load stage")
        sys.exit(1)
    t = time.time()
    n = stage1_load.run(engine, args.input, dataset_name=args.dataset)
    logger.info(f"STAGE 1 LOAD complete — {n} rows inserted ({time.time()-t:.1f}s)")
    return n


def stage_transform(engine, args):
    t = time.time()
    n = stage2_transform.run(engine, dataset_name=args.dataset)
    logger.info(f"STAGE 2 TRANSFORM complete — {n} rows cleaned ({time.time()-t:.1f}s)")
    return n


def stage_enrich(engine, args):
    t = time.time()
    r = stage3_enrich.run(
        engine,
        dataset_name=args.dataset,
        limit=args.limit,
        skip_classification=args.skip_classification,
        skip_aggregates=args.skip_aggregates,
    )
    spam_overridden = r.get("spam_overridden", 0)
    classified = r.get("classified", "skipped")
    logger.info(f"STAGE 3 ENRICH complete — {spam_overridden} spam corrected, {classified} classified ({time.time()-t:.1f}s)")
    return r


def run_all(engine, args):
    logger.info("=" * 60)
    logger.info("FULL PIPELINE RUN")
    logger.info("=" * 60)
    t_total = time.time()

    stage_load(engine, args)
    stage_transform(engine, args)
    stage_enrich(engine, args)

    logger.info(f"PIPELINE COMPLETE — total time: {time.time()-t_total:.1f}s")
    logger.info(f"Database: {args.db}")


def main():
    parser = argparse.ArgumentParser(
        description="Email Intelligence Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with a CSV
  python run_pipeline.py --input data/emails.csv

  # Full run, name the dataset
  python run_pipeline.py --input data/enron.csv --dataset enron

  # Run individual stages
  python run_pipeline.py --stage load      --input data/emails.csv
  python run_pipeline.py --stage transform
  python run_pipeline.py --stage enrich    --limit 100

  # Enrich without re-classifying (just rebuild aggregates)
  python run_pipeline.py --stage enrich --skip-classification
        """
    )
    parser.add_argument("--input",   help="Path to CSV (required for load stage)")
    parser.add_argument("--dataset", help="Dataset name tag (e.g. enron, spamham)")
    parser.add_argument("--db",      default="email_intelligence.db", help="SQLite DB path")
    parser.add_argument("--stage",   choices=["load", "transform", "enrich"],
                        help="Run a single stage (omit for full pipeline)")
    parser.add_argument("--limit",   type=int, help="Max emails to classify (stage 3)")
    parser.add_argument("--skip-classification", action="store_true", dest="skip_classification",
                        help="Skip Claude API calls, only rebuild aggregates")
    parser.add_argument("--skip-aggregates", action="store_true", dest="skip_aggregates",
                        help="Skip rebuilding heavy SQL aggregates (useful for long batch labeling runs)")
    args = parser.parse_args()

    logger.info(f"Initializing DB: {args.db}")
    engine = init_db(args.db)

    if args.stage == "load":
        stage_load(engine, args)
    elif args.stage == "transform":
        stage_transform(engine, args)
    elif args.stage == "enrich":
        stage_enrich(engine, args)
    else:
        run_all(engine, args)


if __name__ == "__main__":
    main()
