"""Command-line interface (CLI) for Smart Waste Image Classification System."""

import argparse
import sys
from pathlib import Path

from src.data.dataset import prepare_dataset
from src.data.preprocessing import get_transforms
from src.evaluation.evaluator import evaluate_model
from src.inference.predictor import Predictor
from src.models.model import build_model, get_device
from src.training.trainer import train_model
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.utils.paths import resolve_path

logger = setup_logger("smart_waste")


def cmd_prepare_data(args, config):
    """Execute dataset download, extraction, validation, and stratified splitting."""
    logger.info("Starting dataset preparation pipeline...")
    metadata = prepare_dataset(config, force=args.force)
    logger.info(f"Dataset preparation completed successfully for {metadata['dataset_name']}.")
    logger.info(f"Total verified samples: {metadata['total_samples']} across {metadata['num_classes']} classes.")


def cmd_train(args, config):
    """Execute model training in full or quick development mode."""
    logger.info("Initializing model training pipeline...")
    train_model(config, quick_mode=args.quick)
    logger.info("Training pipeline finished successfully.")


def cmd_evaluate(args, config):
    """Evaluate model performance on the held-out test split."""
    logger.info("Initializing model evaluation pipeline...")
    metrics = evaluate_model(config)
    logger.info(f"Evaluation complete. Test Accuracy: {metrics['accuracy_pct']}%")


def cmd_predict(args, config):
    """Run inference on a single image and display results."""
    predictor = Predictor(config)
    res = predictor.predict(args.image, top_k=args.top_k)

    print("\nPrediction ----------")
    print(f"Image:           {res['image_path']}")
    print(f"Predicted class: {res['predicted_class']}")
    print(f"Confidence:      {res['confidence']:.2f}%")
    print(f"Model:           {res['model_architecture']}")
    print(f"Device:          {res['device']}")

    if args.top_k > 1:
        print("\nTop predictions:")
        for p in res["top_predictions"]:
            print(f"  {p['rank']}. {p['class']:<10} {p['confidence_pct']:.2f}%")

    if res["low_confidence_warning"]:
        print("\n[WARNING] Model confidence is relatively low (<50%). Consider verifying the prediction manually.")
    print()


def cmd_validate(args, config):
    """Perform self-diagnostic validation check of core system components."""
    print("=" * 65)
    print("  SMART WASTE CLASSIFIER - SYSTEM SELF-DIAGNOSTIC VALIDATION")
    print("=" * 65)

    checks = []

    # 1. Configuration check
    try:
        assert config["project"]["name"] == "smart-waste-classifier"
        checks.append(("Configuration File & Schema", "PASS", "Valid YAML structure and parameters"))
    except Exception as e:
        checks.append(("Configuration File & Schema", "FAIL", str(e)))

    # 2. Directory structure check
    dirs_to_check = [
        config["paths"]["model_dir"],
        config["paths"]["output_dir"],
        config["data"]["raw_dir"],
    ]
    missing_dirs = [d for d in dirs_to_check if not resolve_path(d).parent.exists()]
    if not missing_dirs:
        checks.append(("Directory Hierarchy", "PASS", "All required folders accessible"))
    else:
        checks.append(("Directory Hierarchy", "FAIL", f"Missing parent dirs: {missing_dirs}"))

    # 3. Preprocessing pipeline check
    try:
        import torch
        from PIL import Image
        t_eval = get_transforms(image_size=config["data"]["image_size"], is_training=False)
        dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        dummy_tensor = t_eval(dummy_img)
        expected_shape = (3, config["data"]["image_size"], config["data"]["image_size"])
        assert dummy_tensor.shape == expected_shape, f"Expected {expected_shape}, got {dummy_tensor.shape}"
        checks.append(("Preprocessing Transforms", "PASS", f"Output tensor matches {expected_shape}"))
    except Exception as e:
        checks.append(("Preprocessing Transforms", "FAIL", str(e)))

    # 4. Model Architecture & Forward Pass Check
    try:
        model, device = build_model(config, num_classes=config["data"]["num_classes"])
        dummy_batch = torch.randn(2, 3, config["data"]["image_size"], config["data"]["image_size"]).to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (2, config["data"]["num_classes"])
        checks.append(("Model Instantiation & Forward Pass", "PASS", f"Output shape (2, {config['data']['num_classes']}) on {device.type.upper()}"))
    except Exception as e:
        checks.append(("Model Instantiation & Forward Pass", "FAIL", str(e)))

    # 5. Dataset Split Check
    split_file = resolve_path(config["data"]["split_file"])
    if split_file.is_file():
        checks.append(("Dataset Split Metadata", "PASS", f"Present at {split_file.name}"))
    else:
        checks.append(("Dataset Split Metadata", "INFO", "Not yet generated (run: python -m src.main prepare-data)"))

    # 6. Model Checkpoint Check
    ckpt_file = resolve_path(config["paths"]["best_model_path"])
    if ckpt_file.is_file():
        checks.append(("Trained Model Checkpoint", "PASS", f"Present at {ckpt_file.name}"))
    else:
        checks.append(("Trained Model Checkpoint", "INFO", "Not yet trained (run: python -m src.main train)"))

    # Print summary table
    for component, status, detail in checks:
        color_status = f"[{status}]"
        print(f"  {component:<36} {color_status:<8} {detail}")

    print("=" * 65)
    all_passed = all(c[1] in ("PASS", "INFO") for c in checks)
    if all_passed:
        print("Result: SYSTEM HEALTH VALIDATION SUCCEEDED.")
    else:
        print("Result: SYSTEM HEALTH VALIDATION ENCOUNTERED FAILURES.")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for CLI commands."""
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Smart Waste Image Classification System Using Transfer Learning (MobileNetV3-Small)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available project commands")

    # Command: prepare-data
    parser_prep = subparsers.add_parser("prepare-data", help="Download, validate, and split waste dataset")
    parser_prep.add_argument("--force", action="store_true", help="Force re-download and re-split even if split exists")

    # Command: train
    parser_train = subparsers.add_parser("train", help="Train MobileNetV3-Small classifier on waste dataset")
    parser_train.add_argument("--quick", action="store_true", help="Run rapid development training on a small subset")

    # Command: evaluate
    subparsers.add_parser("evaluate", help="Evaluate trained model strictly on held-out test split")

    # Command: predict
    parser_pred = subparsers.add_parser("predict", help="Perform inference on a single waste image")
    parser_pred.add_argument("--image", type=str, required=True, help="Path to input waste image (.jpg, .jpeg, .png)")
    parser_pred.add_argument("--top-k", type=int, default=1, help="Return top-K probability rankings (default: 1)")

    # Command: validate
    subparsers.add_parser("validate", help="Run complete system health and component integrity check")

    return parser


def main():
    """Main CLI entry point."""
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    command_handlers = {
        "prepare-data": cmd_prepare_data,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "validate": cmd_validate,
    }

    handler = command_handlers.get(args.command)
    if handler:
        try:
            handler(args, config)
        except KeyboardInterrupt:
            logger.warning("Operation aborted by user.")
            sys.exit(130)
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
