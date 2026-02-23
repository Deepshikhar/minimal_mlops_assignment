#!/usr/bin/env python3
"""
run.py

MLOps mini-pipeline:
- Loads config.yaml
- Loads data.csv
- Computes rolling mean on 'close'
- Generates signals (1 if close > rolling_mean else 0)
- Writes metrics.json and logs to the provided log file
- Prints final metrics to stdout

Usage:
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from pandas.errors import EmptyDataError, ParserError

# ---------------------------
# Helper functions
# ---------------------------
def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if logger is re-used
    if not logger.handlers:
        # File handler writes logs to the required path
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def read_config(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Invalid YAML config: {e}")

    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a mapping/object at top level.")

    # Validate required fields
    if "seed" not in cfg or "window" not in cfg or "version" not in cfg:
        raise ValueError("Config missing required fields: 'seed', 'window', 'version'")

    # Type checks
    try:
        seed = int(cfg["seed"])
    except Exception:
        raise ValueError("Config field 'seed' must be an integer")

    try:
        window = int(cfg["window"])
        if window <= 0:
            raise ValueError("window must be positive")
    except Exception:
        raise ValueError("Config field 'window' must be a positive integer")

    version = str(cfg["version"])

    return {"seed": seed, "window": window, "version": version}


def write_metrics(output_path: str, payload: dict):
    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def write_error_json(output_path: str, version: str, message: str):
    payload = {
        "version": version,
        "status": "error",
        "error_message": message,
    }
    # If output_path is provided, try to write; otherwise print to stdout
    if output_path:
        try:
            write_metrics(output_path, payload)
        except Exception:
            # If writing fails, print to stderr
            print(json.dumps(payload, indent=2), file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2), file=sys.stderr)


# ---------------------------
# Main run logic
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="MLOps mini pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV (data.csv)")
    parser.add_argument("--config", required=True, help="Path to config YAML (config.yaml)")
    parser.add_argument("--output", required=True, help="Path to output metrics.json")
    parser.add_argument("--log-file", required=True, help="Path to run.log")
    args = parser.parse_args()

    # For error JSON default version (in case config can't be read)
    default_version = "v1"

    # Ensure log file directory exists
    log_dir = os.path.dirname(os.path.abspath(args.log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger(args.log_file)

    start_time = time.perf_counter()
    job_start_dt = datetime.utcnow().date().isoformat()

    try:
        logger.info(f"Job started {job_start_dt}")
        # Load and validate config
        try:
            cfg = read_config(args.config)
            seed = cfg["seed"]
            window = cfg["window"]
            version = cfg["version"]
        except FileNotFoundError as e:
            # Config missing -> immediate error
            version = default_version
            msg = str(e)
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)
        except ValueError as e:
            # Invalid config format
            version = default_version
            msg = f"Invalid configuration: {e}"
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)
        except Exception as e:
            version = default_version
            msg = f"Error reading configuration: {e}"
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)

        # Log configuration verification
        logger.info(f"Config loaded: seed={seed}, window={window}, version={version}")

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Validate input file
        if not os.path.exists(args.input):
            msg = f"Input file not found: {args.input}"
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)

        # Read CSV
        try:
            df = pd.read_csv(args.input)
        except FileNotFoundError:
            msg = f"Input file not found (race): {args.input}"
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)
        except EmptyDataError:
            msg = "Input CSV is empty."
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)
        except ParserError as e:
            msg = f"Invalid CSV format: {e}"
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)
        except Exception as e:
            msg = f"Failed to read input CSV: {e}"
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)

        # Validate required columns
        if "close" not in df.columns:
            msg = "Required column 'close' is missing from input CSV."
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)

        rows_processed = int(len(df))
        logger.info(f"Data loaded: {rows_processed} rows")

        # Processing: rolling mean on close
        logger.info(f"Computing rolling mean with window={window}")
        # We operate on a copy of the close series to be explicit
        close_series = pd.to_numeric(df["close"], errors="coerce")
        rolling_mean = close_series.rolling(window=window).mean()

        logger.info("Rolling mean calculated")

        # Signal generation: 1 if close > rolling_mean, else 0
        # Handle NaN in rolling_mean by filling with False before astype(int)
        logger.info("Generating signals")
        signals = (close_series > rolling_mean).fillna(False).astype(int)

        # Compute metric
        signal_rate = float(signals.mean()) if len(signals) > 0 else 0.0
        # Round to 4 decimal places as in example
        signal_rate_rounded = round(signal_rate, 4)

        # Compute latency
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Prepare metrics payload
        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": float(f"{signal_rate_rounded:.4f}"),
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success"
        }

        # Write metrics JSON
        try:
            write_metrics(args.output, metrics)
        except Exception as e:
            msg = f"Failed to write metrics JSON: {e}"
            logger.error(msg)
            write_error_json(args.output, version, msg)
            sys.exit(1)

        logger.info(f"Signals generated")
        logger.info(f"Metrics: signal_rate={metrics['value']}, rows_processed={rows_processed}")
        logger.info(f"Job completed successfully in {latency_ms}ms")

        # Print metrics to stdout (required)
        print(json.dumps(metrics, indent=2))

        # Exit success
        sys.exit(0)

    except Exception as e:
        # Catch-all
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.exception("Unhandled exception during run")
        message = f"Unhandled exception: {e}"
        write_error_json(args.output if 'args' in locals() else None, default_version, message)
        sys.exit(1)


if __name__ == "__main__":
    main()