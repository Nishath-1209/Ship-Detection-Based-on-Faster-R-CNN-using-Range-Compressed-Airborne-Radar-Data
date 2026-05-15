import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from dataset import ShipDataset
from faster_rcnn import get_model


def train(img_dir: Path, ann_dir: Path, out_path: Path, epochs: int, batch_size: int, lr: float, subset: int | None):
    ds = ShipDataset(str(img_dir), str(ann_dir))
    if subset is not None:
        ds = Subset(ds, range(subset))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"Epoch {epoch + 1}, Loss: {total:.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=(here.parent / "data" / "images"))
    parser.add_argument("--annotations", type=Path, default=(here.parent / "data" / "annotations"))
    parser.add_argument("--output", type=Path, default=(here.parent / "outputs" / "ship_model.pth"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=10)
    args = parser.parse_args()
    train(args.images, args.annotations, args.output, args.epochs, args.batch_size, args.lr, args.subset)


if __name__ == "__main__":
    main()
