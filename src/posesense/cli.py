"""Command-line interface for PoseSense.

Subcommands (Phase 0 delivers ``check`` and ``download-model``)::

    python -m posesense check               # environment and camera diagnostics
    python -m posesense download-model       # download the MediaPipe model
    python -m posesense live                 # run live recognition (OpenCV HUD)
    python -m posesense gui                  # launch the full GUI
    python -m posesense collect              # collect training data
    python -m posesense dataset              # dataset operations
    python -m posesense train                # train a classifier
    python -m posesense evaluate             # view evaluation results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from posesense import __version__
from posesense.logging_setup import setup_logging

logger = logging.getLogger(__name__)

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="posesense",
        description="PoseSense — Real-time human pose recognition and custom pose trainer",
    )
    parser.add_argument(
        "--version", action="version", version=f"posesense {__version__}"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML configuration file (default: config/default.toml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── check ────────────────────────────────────────────────────────
    sub.add_parser(
        "check",
        help="Check the environment: Python version, library versions, model file",
    )

    # ── download-model ───────────────────────────────────────────────
    dm = sub.add_parser(
        "download-model",
        help="Download the MediaPipe PoseLandmarker model",
    )
    dm.add_argument(
        "--variant",
        choices=["lite", "full", "heavy"],
        default="lite",
        help="Model variant to download (default: lite)",
    )
    dm.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )

    # ── Placeholder subcommands (implemented in later phases) ────────
    sub.add_parser("live", help="Run live pose recognition (OpenCV HUD)")
    vrm_parser = sub.add_parser("vrm", help="Run live VRM avatar tracking")
    vrm_parser.add_argument("--vrm_file", required=True, help="Filename of the .vrm model")
    
    osc_parser = sub.add_parser("osc", help="Broadcast tracking data to external apps (like VSeeFace/Warudo) via VMC Protocol")
    osc_parser.add_argument("--ip", default="127.0.0.1", help="Target IP address for OSC (default: 127.0.0.1)")
    osc_parser.add_argument("--port", type=int, default=39539, help="Target Port for OSC (default: 39539)")
    
    sub.add_parser("gui", help="Launch the full Tkinter GUI")

    collect_parser = sub.add_parser("collect", help="Collect training data")
    collect_parser.add_argument("--label", required=True, help="Pose label to collect")
    collect_parser.add_argument(
        "--burst", type=int, default=0, help="Number of samples in burst mode"
    )

    ds = sub.add_parser("dataset", help="Dataset operations")
    ds.add_argument(
        "action",
        choices=["stats", "export", "import", "validate"],
        help="Dataset action to perform",
    )

    train_parser = sub.add_parser("train", help="Train a pose classifier")
    train_parser.add_argument(
        "--model",
        choices=["rf", "svm", "compare"],
        default="rf",
        help="Classifier type (default: rf)",
    )
    train_parser.add_argument(
        "--activate",
        action="store_true",
        help="Activate the trained model as the live model",
    )

    eval_parser = sub.add_parser("evaluate", help="View evaluation results")
    eval_parser.add_argument("--run", default="latest", help="Run ID to evaluate")

    return parser


def _cmd_check() -> None:
    """Run environment diagnostics."""
    # Import here so missing libraries produce clear messages.
    from posesense.scripts.check_environment import run_checks

    run_checks()


def _cmd_download_model(variant: str, force: bool) -> None:
    """Download the MediaPipe model."""
    from posesense.scripts.download_model import download_model

    download_model(variant=variant, force=force)


def _get_active_model_path(run_id: str) -> Path | None:
    """Resolve the active run ID to a model path."""
    runs_dir = Path("data").parent / "models" / "runs"
    if run_id == "latest":
        if not runs_dir.exists():
            return None
        dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not dirs:
            return None
        return max(dirs, key=lambda d: d.name)
    return runs_dir / run_id

def _prompt_camera(cfg):
    """Prompt the user for camera selection with hardware hints."""
    print("\n--- Camera Selection ---")
    print("Scanning for available cameras (this might take a moment)...")
    import cv2
    import sys
    import subprocess
    
    if sys.platform == "win32":
        try:
            print("\nHardware cameras detected by Windows:")
            cmd = ['powershell', '-Command', "Get-PnpDevice -Class Camera -PresentOnly -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FriendlyName"]
            output = subprocess.check_output(cmd, text=True, timeout=5)
            names = [line.strip() for line in output.split('\n') if line.strip()]
            for name in sorted(set(names)):
                print(f"  - {name}")
            print()
        except Exception:
            pass

    available_cameras = []
    for i in range(5):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append(f"[{i}] Camera {i} (Resolution: {w}x{h})")
            cap.release()
            
    if available_cameras:
        print("Found the following active cameras:")
        for desc in available_cameras:
            print(f"  {desc}")
    else:
        print("  No active cameras detected!")

    cam_input = input(f"\nWhich camera would you like to use? (Press Enter for default {cfg.camera.index}): ").strip()
    if cam_input:
        try:
            import dataclasses
            new_camera = dataclasses.replace(cfg.camera, index=int(cam_input))
            cfg = dataclasses.replace(cfg, camera=new_camera)
        except ValueError:
            print(f"Invalid input. Using default: {cfg.camera.index}")
    print()
    return cfg

def _cmd_live(config_path: Path | None) -> None:
    """Run live pose recognition with the full state machine and pipeline."""
    import time
    import cv2
    import numpy as np

    from posesense.camera.webcam import WebcamSource
    from posesense.config import load_config
    from posesense.features.extractor import FeatureExtractor
    from posesense.model.classifier import PoseClassifier
    from posesense.pose.mediapipe_detector import MediaPipeDetector
    from posesense.pose.quality import QualityGate
    from posesense.runtime.perf import PerformanceMonitor
    from posesense.runtime.state import FrameContext, InferenceStateMachine
    from posesense.ui.overlay import draw_hud, draw_skeleton

    cfg = load_config(config_path)
    cfg = _prompt_camera(cfg)
    logger.info("Initializing live inference pipeline...")

    # Initialize components
    quality_gate = QualityGate(cfg.quality)
    extractor = FeatureExtractor(cfg.features)
    state_machine = InferenceStateMachine(
        debounce_frames=cfg.smoothing.no_person_debounce_frames,
        vote_window=cfg.smoothing.vote_window,
        min_votes=cfg.smoothing.min_votes,
    )
    perf = PerformanceMonitor(
        target_fps=cfg.performance.target_fps,
        log_interval_s=cfg.performance.log_interval_s,
    )

    # Load classifier if available
    classifier = None
    model_path = _get_active_model_path(cfg.inference.active_run_id)
    if model_path and model_path.exists():
        try:
            classifier = PoseClassifier(
                model_path,
                min_confidence=cfg.rejection.min_confidence,
                min_margin=cfg.rejection.min_margin,
            )
            logger.info("Loaded active model from %s", model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
    else:
        logger.warning("No active model found (run_id=%s)", cfg.inference.active_run_id)

    try:
        with WebcamSource(cfg.camera) as cam, MediaPipeDetector(cfg.detector) as detector:
            window_name = "PoseSense Live"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            logger.info("Starting live view. Press 'q' or 'ESC' to quit, 'd' to toggle debug.")
            debug_mode = False

            while True:
                perf.mark_frame_start()
                ctx = FrameContext(has_model=(classifier is not None))

                # ── Capture ──
                perf.start_stage()
                try:
                    frame = cam.read()
                except Exception as e:
                    logger.error("Camera error: %s", e)
                    ctx.camera_failed = True
                    frame = np.zeros((cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8)
                    frame[:] = (0, 0, 100)
                perf.end_stage("capture")

                if cfg.camera.mirror_display and not ctx.camera_failed:
                    display_frame = cv2.flip(frame, 1)
                else:
                    display_frame = frame.copy()

                pose_result = None
                
                if not ctx.camera_failed:
                    # ── Detection ──
                    perf.start_stage()
                    timestamp_ms = int(time.monotonic_ns() // 1_000_000)
                    pose_result = detector.process_frame(frame, timestamp_ms)
                    perf.end_stage("detect")

                    if pose_result is not None:
                        ctx.has_landmarks = True
                        
                        # ── Quality Gate ──
                        report = quality_gate.evaluate(pose_result)
                        if not report.ok:
                            ctx.quality_reasons = report.reasons
                            
                        if classifier is not None:
                            # ── Feature Extraction ──
                            perf.start_stage()
                            features = extractor.extract(pose_result)
                            perf.end_stage("features")

                            # ── Classification ──
                            perf.start_stage()
                            label, score = classifier.predict(features)
                            ctx.raw_predicted_label = label
                            ctx.raw_predicted_score = score
                            perf.end_stage("classify")
                
                # ── State Machine ──
                state_out = state_machine.evaluate(ctx)

                # ── Drawing ──
                perf.start_stage()
                if pose_result:
                    if cfg.camera.mirror_display:
                        import dataclasses
                        mirrored_lms = []
                        for lm in pose_result.landmarks:
                            mirrored_lms.append(dataclasses.replace(lm, x=1.0 - lm.x))
                        display_pose_result = dataclasses.replace(pose_result, landmarks=tuple(mirrored_lms))
                    else:
                        display_pose_result = pose_result

                    draw_skeleton(display_frame, display_pose_result, debug=debug_mode)

                draw_hud(
                    display_frame,
                    state_out.state,
                    fps=perf.fps,
                    reasons=state_out.quality_reasons,
                    label=state_out.smoothed_label,
                    confidence=state_out.smoothed_score,
                )
                perf.end_stage("draw")

                cv2.imshow(window_name, display_frame)

                perf.log_summary_if_needed()

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("d"):
                    debug_mode = not debug_mode

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()


def _cmd_vrm(config_path: Path | None, vrm_file: str) -> None:
    """Run live pose tracking and send data to the VRM web viewer."""
    import time
    import cv2
    import webbrowser

    from posesense.camera.webcam import WebcamSource
    from posesense.config import load_config
    from posesense.pose.mediapipe_detector import MediaPipeDetector
    from posesense.vrm_server import start_vrm_server, SSEHandler
    
    cfg = load_config(config_path)
    cfg = _prompt_camera(cfg)
    logger.info("Initializing VRM pipeline...")

    # Start the server
    port = 8080
    start_vrm_server(vrm_file, port)
    logger.info(f"VRM Server started at http://localhost:{port}")
    
    # Open the browser
    import time
    webbrowser.open(f"http://localhost:{port}/?t={int(time.time())}")

    try:
        with WebcamSource(cfg.camera) as cam, MediaPipeDetector(cfg.detector) as detector:
            window_name = "PoseSense VRM Tracker"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while True:
                try:
                    frame = cam.read()
                except Exception as e:
                    logger.error("Camera error: %s", e)
                    break
                    
                display_frame = cv2.flip(frame, 1) if cfg.camera.mirror_display else frame.copy()

                timestamp_ms = int(time.monotonic_ns() // 1_000_000)
                pose_result = detector.process_frame(frame, timestamp_ms)

                if pose_result:
                    # Send raw, un-mirrored coordinates to Kalidokit to prevent bone twisting
                    raw_world = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in (pose_result.world_landmarks or [])]
                    raw_norm = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in pose_result.landmarks]
                    
                    SSEHandler.latest_pose = {
                        "world": raw_world,
                        "norm": raw_norm
                    }
                    
                    if cfg.camera.mirror_display:
                        import dataclasses
                        display_pose_result = dataclasses.replace(
                            pose_result, 
                            landmarks=tuple(dataclasses.replace(lm, x=1.0 - lm.x) for lm in pose_result.landmarks)
                        )
                    else:
                        display_pose_result = pose_result
                        
                    from posesense.ui.overlay import draw_skeleton
                    draw_skeleton(display_frame, display_pose_result, debug=False)

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()


def _cmd_osc(config_path: Path | None, ip: str, port: int) -> None:
    """Run live pose tracking and broadcast data via VMC/OSC to an external app."""
    import time
    import cv2

    from posesense.camera.webcam import WebcamSource
    from posesense.config import load_config
    from posesense.pose.mediapipe_detector import MediaPipeDetector
    from posesense.vmc import VMCSender
    
    cfg = load_config(config_path)
    cfg = _prompt_camera(cfg)
    logger.info(f"Initializing VMC OSC Broadcaster to {ip}:{port}...")

    sender = VMCSender(ip, port)

    try:
        with WebcamSource(cfg.camera) as cam, MediaPipeDetector(cfg.detector) as detector:
            window_name = "PoseSense VMC Broadcaster"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while True:
                try:
                    frame = cam.read()
                except Exception as e:
                    logger.error("Camera error: %s", e)
                    break
                    
                display_frame = cv2.flip(frame, 1) if cfg.camera.mirror_display else frame.copy()

                timestamp_ms = int(time.monotonic_ns() // 1_000_000)
                pose_result = detector.process_frame(frame, timestamp_ms)

                if pose_result:
                    sender.broadcast_pose(pose_result)
                    
                    from posesense.ui.overlay import draw_skeleton
                    if cfg.camera.mirror_display:
                        import dataclasses
                        display_pose_result = dataclasses.replace(
                            pose_result, 
                            landmarks=tuple(dataclasses.replace(lm, x=1.0 - lm.x) for lm in pose_result.landmarks)
                        )
                    else:
                        display_pose_result = pose_result
                        
                    draw_skeleton(display_frame, display_pose_result, debug=False)

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()


def _cmd_collect(config_path: Path | None, label: str, burst: int) -> None:
    """Run data collection mode (Phase 2 UI)."""
    import time

    import cv2
    import numpy as np

    from posesense.camera.webcam import WebcamSource
    from posesense.config import load_config
    from posesense.data.dataset import PoseDataset
    from posesense.features.extractor import FeatureExtractor
    from posesense.pose.mediapipe_detector import MediaPipeDetector
    from posesense.pose.quality import QualityGate
    from posesense.types import AppState
    from posesense.ui.overlay import draw_hud, draw_skeleton

    cfg = load_config(config_path)
    cfg = _prompt_camera(cfg)
    dataset = PoseDataset()
    quality_gate = QualityGate(cfg.quality)
    extractor = FeatureExtractor(cfg.features)
    feature_names = extractor.get_feature_names()

    session_id = dataset.create_session_id()
    logger.info("Starting collection session %s for label '%s'", session_id, label)

    samples_collected = 0
    samples_to_collect = burst if burst > 0 else float("inf")

    try:
        with WebcamSource(cfg.camera) as cam, MediaPipeDetector(cfg.detector) as detector:
            window_name = f"PoseSense Collect: {label}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            logger.info("Press 's' to capture a sample, 'c' to toggle continuous burst, 'q' to quit.")

            continuous_mode = False
            last_time = time.monotonic()
            fps_alpha = 0.1
            fps = 0.0

            while samples_collected < samples_to_collect:
                try:
                    frame = cam.read()
                    state = AppState.NO_MODEL
                except Exception as e:
                    logger.error("Camera error: %s", e)
                    frame = np.zeros((cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8)
                    frame[:] = (0, 0, 100)
                    state = AppState.CAMERA_UNAVAILABLE
                    draw_hud(frame, state, fps=0.0)
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1000) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue

                if cfg.camera.mirror_display:
                    display_frame = cv2.flip(frame, 1)
                else:
                    display_frame = frame.copy()

                timestamp_ms = int(time.monotonic_ns() // 1_000_000)
                pose_result = detector.process_frame(frame, timestamp_ms)

                report = None
                if pose_result is None:
                    state = AppState.NO_PERSON
                else:
                    report = quality_gate.evaluate(pose_result)
                    if not report.ok:
                        state = AppState.LOW_QUALITY
                    else:
                        state = AppState.NO_MODEL  # In collect mode, we just mean "Valid pose, no classifier yet"

                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("c"):
                    continuous_mode = not continuous_mode
                    logger.info("Continuous mode: %s", "ON" if continuous_mode else "OFF")

                # Collect sample if requested (and quality is good)
                trigger = (key == ord("s")) or continuous_mode

                if trigger and pose_result and report and report.ok:
                    features = extractor.extract(pose_result)
                    features["timestamp"] = timestamp_ms

                    dataset.append_samples(label, session_id, feature_names, [features])
                    samples_collected += 1

                    # Flash the screen green to indicate successful capture
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (cfg.camera.width, cfg.camera.height), (0, 255, 0), 10)
                    cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                    logger.info("Captured sample %d/%s", samples_collected, burst if burst > 0 else "inf")

                # Drawing
                if pose_result:
                    if cfg.camera.mirror_display:
                        import dataclasses
                        mirrored_lms = []
                        for lm in pose_result.landmarks:
                            mirrored_lms.append(dataclasses.replace(lm, x=1.0 - lm.x))
                        display_pose_result = dataclasses.replace(pose_result, landmarks=tuple(mirrored_lms))
                    else:
                        display_pose_result = pose_result

                    draw_skeleton(display_frame, display_pose_result, debug=False)

                now = time.monotonic()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    current_fps = 1.0 / dt
                    fps = (current_fps * fps_alpha) + (fps * (1.0 - fps_alpha))

                reasons = report.reasons if report else None
                draw_hud(display_frame, state, fps=fps, reasons=reasons)

                # Add collection progress text
                progress_text = f"Collected: {samples_collected}"
                if burst > 0:
                    progress_text += f" / {burst}"
                if continuous_mode:
                    progress_text += " [BURST]"
                cv2.putText(display_frame, progress_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow(window_name, display_frame)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()


def _cmd_dataset(config_path: Path | None, action: str) -> None:
    """Run dataset operations (Phase 2 UI)."""
    from posesense.config import load_config
    from posesense.data.dataset import PoseDataset
    from posesense.features.extractor import FeatureExtractor
    
    cfg = load_config(config_path)
    dataset = PoseDataset()
    extractor = FeatureExtractor(cfg.features)
    
    if action == "stats":
        stats = dataset.get_stats()
        if not stats:
            print("Dataset is empty.")
            return
        
        print(f"{'Label':<20} | {'Samples':<10} | {'Sessions':<10}")
        print("-" * 46)
        total_samples = 0
        total_sessions = 0
        for label, st in sorted(stats.items()):
            print(f"{label:<20} | {st['samples']:<10} | {st['sessions']:<10}")
            total_samples += st["samples"]
            total_sessions += st["sessions"]
        print("-" * 46)
        print(f"{'Total':<20} | {total_samples:<10} | {total_sessions:<10}")
        
    elif action == "export":
        out_path = Path("dataset_export.csv")
        dataset.export_dataset(out_path)
        print(f"Exported dataset to {out_path.absolute()}")
        
    elif action == "import":
        in_path = Path("dataset_export.csv")
        if not in_path.exists():
            print(f"Import file {in_path} not found.")
            import sys
            sys.exit(1)
        dataset.import_dataset(in_path)
        print(f"Imported dataset from {in_path.absolute()}")
        
    elif action == "validate":
        expected_features = extractor.get_feature_names()
        errors = dataset.validate_dataset(expected_features)
        if errors:
            print("Dataset validation failed:")
            for err in errors:
                print(f"  - {err}")
            import sys
            sys.exit(1)
        else:
            print("Dataset is valid.")


def _cmd_train(config_path: Path | None, model_type: str, activate: bool) -> None:
    """Train a pose classifier."""
    from posesense.config import load_config
    from posesense.data.dataset import PoseDataset
    from posesense.features.extractor import FeatureExtractor
    from posesense.model.trainer import Trainer
    
    cfg = load_config(config_path)
    dataset = PoseDataset()
    extractor = FeatureExtractor(cfg.features)
    
    trainer = Trainer(dataset, extractor)
    print(f"Training {model_type} model...")
    run_dir = trainer.train(model_type=model_type)
    
    if run_dir:
        print(f"Model saved to: {run_dir}")
        if activate:
            from posesense.config import update_active_model
            update_active_model(run_dir.name, config_path)
            print(f"Activated model {run_dir.name} in config.")
    else:
        print("Training failed.")
        import sys
        sys.exit(1)


def _cmd_evaluate(config_path: Path | None, run: str) -> None:
    """Evaluate a trained model artifact."""
    from posesense.model.artifact import ModelArtifact
    import json
    
    # Very basic placeholder evaluation readout
    run_dir = Path("models/runs") / run
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return
        
    meta_path = run_dir / ModelArtifact.METADATA_FILENAME
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            metrics = metadata.get("metrics", {})
            print(f"Metrics for run {run}:")
            print(f"  Macro F1: {metrics.get('macro_f1', 0):.3f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
            if "weakest_class" in metrics:
                wc = metrics["weakest_class"]
                print(f"  Weakest class: {wc['label']} (F1: {wc['f1']:.3f})")
    else:
        print("No metadata found.")


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the PoseSense CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level, log_to_file=False)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "check":
        _cmd_check()
    elif args.command == "download-model":
        _cmd_download_model(variant=args.variant, force=args.force)
    elif args.command == "live":
        _cmd_live(args.config)
    elif args.command == "vrm":
        _cmd_vrm(args.config, args.vrm_file)
    elif args.command == "osc":
        _cmd_osc(args.config, args.ip, args.port)
    elif args.command == "collect":
        _cmd_collect(args.config, args.label, args.burst)
    elif args.command == "dataset":
        _cmd_dataset(args.config, args.action)
    elif args.command == "train":
        _cmd_train(args.config, args.model, args.activate)
    elif args.command == "evaluate":
        _cmd_evaluate(args.config, args.run)
    elif args.command == "gui":
        print(f"Command '{args.command}' will be implemented in a later phase.")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)
