import argparse
from pathlib import Path

from config import MODELS_DIR


USE_OPTICAL_FLOW = False  # True: Farneback False: absdiff
MIN_MOTION_PIXELS = 2.0
MIN_LUMA_DIFFERENCE = 25
MIN_CONTOUR_AREA = 10000


def calculate_motion(previous_gray, gray):
    import cv2
    import numpy as np

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray, gray, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = (magnitude > MIN_MOTION_PIXELS).astype(np.uint8) * 255
    hsv = np.zeros((*gray.shape, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 90 / np.pi
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude * 25, 0, 255).astype(np.uint8)
    flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return mask, flow_image


def classify_boxes(frame, boxes, classifier):
    """Return (clipped box, label, score) for each valid BGR frame crop."""
    import cv2
    from PIL import Image

    height, width = frame.shape[:2]
    predictions = []
    for x, y, w, h in boxes:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        label, score = classifier.predict_image(Image.fromarray(crop))[0]
        predictions.append(((x1, y1, x2 - x1, y2 - y1), label, score))
    return predictions


def run_camera(classifier=None, camera_index=0):
    print("Lade OpenCV und NumPy ...", flush=True)
    import cv2
    import numpy as np

    print(f"Öffne Kamera {camera_index} ...", flush=True)
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            print("Kamera konnte nicht geöffnet werden. Kameraberechtigung und --camera-index prüfen.", flush=True)
            return

        print("Kamera geöffnet. Lese erstes Bild ...", flush=True)
        success, frame = cap.read()
        if not success:
            print("Could not read initial camera frame")
            return

        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return
        print("Kamera läuft. Mit q beenden.", flush=True)

        while True:
            success, frame = cap.read()
            if not success:
                print("Could not read camera")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape != previous_gray.shape:
                previous_gray = gray
                continue

            if USE_OPTICAL_FLOW:
                mask, flow_image = calculate_motion(previous_gray, gray)
                cv2.imshow("Optical Flow", flow_image)
            else:
                difference = cv2.absdiff(previous_gray, gray)
                mask = (difference >= MIN_LUMA_DIFFERENCE).astype(np.uint8) * 255
                cv2.imshow("Frame Difference", difference)

            previous_gray = gray

            # Apply the same cleanup and contour detection to both methods.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            boxes = []
            scores = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, w, h])
                scores.append(float(area))

            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                score_threshold=0.0,
                nms_threshold=0.3,
            )

            selected_boxes = [boxes[index] for index in np.asarray(indices).reshape(-1)]
            # Classify all crops before drawing so annotations never enter the model.
            predictions = (classify_boxes(frame, selected_boxes, classifier)
                           if classifier is not None else
                           [(box, "Bewegung", None) for box in selected_boxes])
            for (x, y, w, h), label, score in predictions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label} ({score:.1%})" if score is not None else label,
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Camera", frame)
            cv2.imshow("Motion Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bewegte Bildausschnitte mit AlexNet klassifizieren")
    default_model = MODELS_DIR / "AlexNet.pt"
    if not default_model.is_file():
        default_model = MODELS_DIR / "AlexNet.onnx"
    parser.add_argument("--model", type=Path, default=default_model,
                        help="Trainierter .pt-Checkpoint oder exportiertes .onnx-Modell")
    parser.add_argument("--labels", type=Path, help="ONNX: Standard <Modellname>.labels.txt")
    parser.add_argument("--image-size", type=int, help="ONNX: Bildgröße bei dynamischer Eingabe")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--motion-only", action="store_true",
                        help="Nur Kamera und Bewegungsboxen testen, ohne Modell")
    parser.add_argument("--imagenet", action="store_true",
                        help="ONNX: Zuschnitt und Normalisierung für vortrainiertes ImageNet-AlexNet")
    args = parser.parse_args(argv)
    if args.motion_only:
        run_camera(camera_index=args.camera_index)
        return
    if not args.model.is_file():
        parser.exit(1, f"Modell nicht gefunden: {args.model}\n"
                    "Bitte einen trainierten AlexNet.pt-Checkpoint in models/ ablegen "
                    "oder mit --model angeben.\n")
    if args.model.suffix.lower() not in {".pt", ".onnx"}:
        parser.error("--model muss eine .pt- oder .onnx-Datei sein")
    if args.model.suffix.lower() == ".pt" and (args.labels or args.image_size is not None or args.imagenet):
        parser.error("--labels, --image-size und --imagenet gelten nur für ONNX; .pt enthält die Trainingsangaben")
    if args.model.suffix.lower() == ".onnx":
        labels_path = args.labels or args.model.with_suffix(".labels.txt")
        if not labels_path.is_file():
            parser.exit(1, f"Label-Datei fehlt: {labels_path}\n"
                        "Die beim Export erzeugte .labels.txt neben das Modell legen "
                        "oder mit --labels angeben.\n"
                        "Kamera ohne Modell testen: --motion-only\n")
    print(f"Lade Modell und Inferenz-Bibliotheken: {args.model} ...", flush=True)
    from inference import CheckpointClassifier, OnnxClassifier
    inference_errors = (OSError, ValueError, KeyError, RuntimeError)
    if args.model.suffix.lower() == ".onnx":
        from onnxruntime.capi.onnxruntime_pybind11_state import (
            Fail, InvalidArgument, InvalidGraph, NoSuchFile,
        )
        inference_errors += (Fail, InvalidArgument, InvalidGraph, NoSuchFile)
    try:
        if args.model.suffix.lower() == ".pt":
            classifier = CheckpointClassifier(args.model)
        else:
            classifier = OnnxClassifier(args.model, args.labels, args.image_size, imagenet=args.imagenet)
    except inference_errors as error:
        parser.exit(1, f"Modell konnte nicht geladen werden: {error}\n")
    print(f"Modell geladen: {args.model}. Bewegtes Objekt zeigen; q beendet die Kamera.", flush=True)
    run_camera(classifier, args.camera_index)


if __name__ == "__main__":
    main()
