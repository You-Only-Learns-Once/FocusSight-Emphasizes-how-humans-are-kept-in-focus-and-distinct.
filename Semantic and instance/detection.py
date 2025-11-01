# ==============================================
# YOLOv8 Semantic + Instance Segmentation
# Project: Happy New Year
# ==============================================
import cv2
import torch
import argparse
import os
from ultralytics import YOLO

# -------------------------------
# SETUP ARGUMENTS
# -------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Semantic + Instance Segmentation")
    parser.add_argument("--source", type=str, default="0", help="Path to image/video or '0' for webcam")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="YOLOv8 segmentation model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save the segmentation output")
    parser.add_argument("--save_dir", type=str, default="seg_results", help="Directory to save results")
    return parser.parse_args()


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    args = get_args()

    # Choose GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device.upper()}")

    # Load YOLOv8 segmentation model
    model = YOLO(args.model)
    model.to(device)
    print(f"✅ Model '{args.model}' loaded successfully")

    # Create save directory if required
    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)

    # If source = 0, open webcam
    if args.source == "0":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access webcam.")
            return

        print("🎥 Running segmentation on webcam feed (press 'q' to quit)")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO segmentation
            results = model.predict(frame, conf=args.conf, device=device, verbose=False)
            annotated_frame = results[0].plot()  # Draw masks + boxes + labels

            # Show live segmented frame
            cv2.imshow("YOLOv8 Segmentation - Happy New Year", annotated_frame)

            # Save frame if requested
            if args.save:
                cv2.imwrite(os.path.join(args.save_dir, "webcam_segmented.jpg"), annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Run on image or video
        print(f"🖼️ Processing: {args.source}")
        results = model.predict(source=args.source, conf=args.conf, device=device, show=True, save=args.save, project=args.save_dir)
        print("✅ Segmentation completed.")

    print("🎉 Done! — Happy New Year Project Complete")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()
