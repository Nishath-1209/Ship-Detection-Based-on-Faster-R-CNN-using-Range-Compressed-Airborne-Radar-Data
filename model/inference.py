import os
import argparse
from pathlib import Path
import cv2
import torch
import torchvision
from faster_rcnn import get_model


def load_model(weights_path, device):
    m = get_model()
    state = torch.load(str(weights_path), map_location=device)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m


def run_inference(input_dir, output_dir, weights_path, score_thresh, nms_iou):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("🔍 Running inference on new images...\n")
    exts = (".png", ".jpg", ".jpeg")
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"❌ Unable to read {p.name}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb / 255.0).permute(2, 0, 1).float().to(device)
        with torch.no_grad():
            pred = model([img_tensor])[0]
        keep = torchvision.ops.nms(pred["boxes"], pred["scores"], nms_iou)
        boxes = pred["boxes"][keep]
        scores = pred["scores"][keep]
        ship_count = 0
        for box, score in zip(boxes, scores):
            if float(score) > score_thresh:
                ship_count += 1
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_rgb, f"Ship {float(score):.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out_path = output_dir / p.name
        cv2.imwrite(str(out_path), img_bgr)
        print(f"Saved detection to {out_path} (Detected {ship_count} ships)")
    print("\n✅ Inference completed.")


def main():
    here = Path(__file__).resolve().parent
    default_input = (here.parent / "test_inputs").resolve()
    default_output = (here.parent / "outputs").resolve()
    default_weights = (here.parent / "outputs" / "ship_model.pth").resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--weights", type=Path, default=default_weights)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.2)
    args = parser.parse_args()
    run_inference(args.input, args.output, args.weights, args.score_threshold, args.nms_iou)


if __name__ == "__main__":
    main()
