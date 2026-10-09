"""Camera-only head/iris benchmark. No Furhat, audio, or application FPS cap."""
import argparse
import csv
import json
from datetime import datetime
from math import asin, atan2, degrees
from pathlib import Path
from time import perf_counter, monotonic_ns
from urllib.request import urlopen
from zipfile import is_zipfile

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/face_landmarker.task"
URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


def head_rotation(matrix):
    """Remove scale/shear from the face transform, retaining a proper rotation."""
    block = np.asarray(matrix, dtype=float)[:3, :3]
    if block.shape != (3, 3) or not np.isfinite(block).all():
        raise ValueError("Invalid face matrix")
    u, s, vt = np.linalg.svd(block)
    if s.min() < 1e-6:
        raise ValueError("Degenerate face matrix")
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def head_angles(r):
    """Yaw, pitch, roll in degrees; R = Rz(roll) Ry(yaw) Rx(pitch)."""
    yaw = asin(float(np.clip(-r[2, 0], -1, 1)))
    pitch = atan2(r[2, 1], r[2, 2])
    roll = atan2(r[1, 0], r[0, 0])
    if abs(np.cos(yaw)) < 1e-6:
        pitch, roll = atan2(-r[1, 2], r[1, 1]), 0
    return tuple(float(degrees(x)) for x in (yaw, pitch, roll))


def iris_offsets(points):
    """Eye-relative iris displacement, NOT calibrated gaze angles.

    Coordinates use pixels to account for the image's aspect ratio. Closed or
    tiny eyes are skipped; eyelid geometry is a heuristic, not a blink model.
    """
    measurements, eyes = [], []
    for iris, corner1, corner2, upper, lower in (
        (468, 33, 133, 159, 145), (473, 362, 263, 386, 374)
    ):
        a, b = sorted((points[corner1], points[corner2]), key=lambda p: p[0])
        width = np.linalg.norm(b - a)
        if width < 10:
            continue
        horizontal = (b - a) / width
        vertical = np.array([-horizontal[1], horizontal[0]])
        opening = abs(np.dot(points[lower] - points[upper], vertical)) / width
        if opening < 0.10:
            continue
        delta = points[iris] - (a + b) / 2
        uv = np.array([np.dot(delta, horizontal), np.dot(delta, vertical)]) / width
        if abs(uv[0]) < 0.6 and abs(uv[1]) < 0.4:
            measurements.append(uv)
            eyes.append((points[iris], horizontal, vertical))
    return (np.mean(measurements, axis=0) if measurements else None), eyes


def load_model(task, faces):
    if not MODEL.exists():
        print("[DOWNLOAD] Face Landmarker once; subsequent runs reuse the cache.")
        MODEL.parent.mkdir(exist_ok=True)
        temporary = MODEL.with_suffix(".download")
        try:
            with urlopen(URL, timeout=60) as response:
                temporary.write_bytes(response.read())
            if not is_zipfile(temporary):
                raise RuntimeError("Downloaded file is not a Face Landmarker task bundle")
            temporary.replace(MODEL)
        finally:
            temporary.unlink(missing_ok=True)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=faces,
        output_facial_transformation_matrixes=task != "gaze",
        output_face_blendshapes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("head", "gaze", "both"), default="both")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640, help="Camera request and processing width")
    parser.add_argument("--faces", type=int, default=1, help="Maximum faces; slots are not persistent IDs")
    parser.add_argument("--seconds", type=float, default=0, help="0 runs until Q/Esc")
    parser.add_argument("--landmarks", action="store_true", help="Draw all 478 landmarks")
    parser.add_argument("--mirror", action="store_true", help="Mirror input before estimation/display")
    args = parser.parse_args()
    if args.width < 160 or args.faces < 1 or args.seconds < 0:
        parser.error("width must be >=160, faces >=1, seconds >=0")
    output = ROOT / "vision_output/head_gaze_compare" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    keys = ["capture_ms", "prepare_ms", "landmarker_ms", "geometry_ms", "display_ms", "total_ms", "faces"]
    rows, neutral = [], {}
    cap = model = None
    started = perf_counter()
    try:
        model = load_model(args.task, args.faces)
        load_seconds = perf_counter() - started
        print(f"[MODEL] Face Landmarker | CPU | task={args.task} | weights={MODEL.stat().st_size / 1e6:.2f}MB | load={load_seconds:.2f}s")
        print("[MEASURE] One shared face model. Head/iris geometry timed separately. First 5 frames excluded.")
        print("[KEYS] Q/Esc quits; C sets neutral head/iris offsets for visible faces; R resets. No Furhat.")
        print(f"[OUTPUT] {output}")
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if not cap.isOpened():
            raise RuntimeError("Camera could not be opened")
        frame_number, timestamp, last_log = 0, -1, perf_counter()
        run_started = perf_counter()
        with (output / "timings.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", *keys])
            writer.writeheader()
            while args.seconds == 0 or perf_counter() - run_started < args.seconds:
                begin = perf_counter()
                ok, frame = cap.read()
                captured = perf_counter()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                if frame.shape[1] != args.width:
                    frame = cv2.resize(frame, (args.width, round(frame.shape[0] * args.width / frame.shape[1])))
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(timestamp + 1, monotonic_ns() // 1_000_000)
                prepared = perf_counter()
                result = model.detect_for_video(image, timestamp)
                inferred = perf_counter()
                observations = []
                h, w = frame.shape[:2]
                for slot, landmarks in enumerate(result.face_landmarks):
                    points = np.array([(p.x * w, p.y * h) for p in landmarks])
                    rotation = head_rotation(result.facial_transformation_matrixes[slot]) if args.task != "gaze" else None
                    gaze, eyes = iris_offsets(points) if args.task != "head" else (None, [])
                    reference = neutral.get(slot, (None, None))
                    angles = head_angles(reference[0].T @ rotation if reference[0] is not None else rotation) if rotation is not None else None
                    offset = gaze - reference[1] if gaze is not None and reference[1] is not None else gaze
                    observations.append((points, rotation, gaze, angles, offset, eyes))
                # Clear neutral references when no faces are visible; slots are not identities.
                if not observations:
                    neutral.clear()
                geometric = perf_counter()
                for slot, (points, rotation, gaze, angles, offset, eyes) in enumerate(observations):
                    if args.landmarks:
                        for p in points:
                            cv2.circle(frame, tuple(p.astype(int)), 1, (180, 180, 180), -1)
                    low, high = points.min(axis=0).astype(int), points.max(axis=0).astype(int)
                    cv2.rectangle(frame, tuple(low), tuple(high), (80, 200, 80), 1)
                    lines = [f"Face slot {slot + 1}"]
                    if angles is not None:
                        lines.append("Head yaw/pitch/roll: " + "/".join(f"{a:+.0f}" for a in angles) + " deg")
                        origin = points[1].astype(int)
                        for axis, color in zip(rotation.T, ((0, 0, 255), (0, 255, 0), (255, 100, 0))):
                            end = origin + (np.array([axis[0], -axis[1]]) * 65).astype(int)
                            cv2.arrowedLine(frame, tuple(origin), tuple(end), color, 2)
                    if args.task != "head":
                        lines.append("Iris: eyes closed/unreliable" if offset is None else f"Iris u={offset[0]:+.3f} v={offset[1]:+.3f} (relative)")
                        if offset is not None:
                            for center, horizontal, vertical in eyes:
                                end = center + 240 * (offset[0] * horizontal + offset[1] * vertical)
                                cv2.arrowedLine(frame, tuple(center.astype(int)), tuple(end.astype(int)), (0, 230, 255), 2)
                    for line_index, line in enumerate(lines):
                        cv2.putText(frame, line, (10, 65 + slot * 85 + line_index * 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                recent = rows[-30:]
                fps = 1000 / np.mean([r["total_ms"] for r in recent]) if recent else 0
                cv2.putText(frame, f"CPU | {args.task} | {fps:.1f} FPS | C: neutral  R: reset  Q: exit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 1)
                cv2.imshow("Head and iris benchmark", frame)
                key = cv2.waitKey(1) & 0xFF
                ended = perf_counter()
                frame_number += 1
                values = [captured-begin, prepared-captured, inferred-prepared, geometric-inferred, ended-geometric, ended-begin]
                row = dict(zip(keys, [*(v * 1000 for v in values), len(observations)]))
                if frame_number > 5:
                    rows.append(row)
                    writer.writerow({"frame": frame_number, **row})
                if ended - last_log >= 2 and rows:
                    means = {k: np.mean([r[k] for r in rows[-60:]]) for k in keys}
                    print(f"[TIMING] {1000/means['total_ms']:.2f} FPS | face model={means['landmarker_ms']:.2f}ms | geometry={means['geometry_ms']:.2f}ms | capture={means['capture_ms']:.2f}ms | prepare={means['prepare_ms']:.2f}ms | display={means['display_ms']:.2f}ms | faces={len(observations)}")
                    handle.flush()
                    last_log = ended
                if key in (27, ord('q')):
                    break
                if key == ord('c'):
                    neutral = {i: (o[1].copy() if o[1] is not None else None, o[2].copy() if o[2] is not None else None) for i, o in enumerate(observations)}
                    print("[NEUTRAL] Reference set for visible face slots; reset after changing people.")
                if key == ord('r'):
                    neutral.clear()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if cap is not None:
            cap.release()
        if model is not None:
            model.close()
        cv2.destroyAllWindows()
        summary = {"settings": vars(args), "backend": "CPU", "frames_measured": len(rows), "warmup_frames_excluded": 5}
        if rows:
            summary["fps"] = 1000 / float(np.mean([r["total_ms"] for r in rows]))
            summary["timing_ms"] = {k: {"mean": float(np.mean([r[k] for r in rows])), "p95": float(np.percentile([r[k] for r in rows], 95))} for k in keys if k != "faces"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[SAVED] {output}")


if __name__ == "__main__":
    main()
