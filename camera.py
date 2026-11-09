import cv2
import numpy as np


USE_OPTICAL_FLOW = False  # True: Farneback False: absdiff
MIN_MOTION_PIXELS = 2.0
MIN_LUMA_DIFFERENCE = 25
MIN_CONTOUR_AREA = 10000


def calculate_motion(previous_gray, gray):
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


def main():
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            print("Could not open camera")
            return

        success, frame = cap.read()
        if not success:
            print("Could not read initial camera frame")
            return

        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

            for index in np.asarray(indices).reshape(-1):
                x, y, w, h = boxes[index]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Camera", frame)
            cv2.imshow("Motion Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
