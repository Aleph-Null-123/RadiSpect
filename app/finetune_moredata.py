import json, math, time
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ConvSAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,16,3,2,1), nn.ReLU(inplace=True),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(inplace=True),
        )
        self.enc_lin = nn.Linear(64*28*28, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 64*28*28)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,1, 4,2,1), nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return F.relu(self.enc_lin(h))

    def decode(self, z):
        h = self.dec_lin(z).view(-1,64,28,28)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x); xhat = self.decode(z)
        return xhat, z

class XRFolder(Dataset):
    def __init__(self, paths, img_size=224):
        self.paths = list(paths)
        self.img_size = img_size
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("L").resize((self.img_size, self.img_size), Image.LANCZOS)
        x = (np.asarray(img, dtype=np.float32)/255.0)[None, ...]  # (1,H,W)
        return torch.from_numpy(x), p

@torch.no_grad()
def eval_val(model, val_loader, lambda_l1):
    model.eval()
    tot_mse = 0.0; tot_n = 0; actives = []
    for x,_ in val_loader:
        x = x.to("cpu")
        xhat, z = model(x)
        mse = F.mse_loss(xhat, x, reduction="sum").item()
        tot_mse += mse
        tot_n += x.numel()
        actives.append((z>1e-6).float().sum(dim=1).mean().item())
    model.train()
    return tot_mse/tot_n, float(np.mean(actives) if actives else 0.0)

def train_more(
    from_run: str,
    train_glob: str = "data/images_normalized/*.png",
    save_to: str | None = None,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_lambda: float | None = None,
):
    from_run = Path(from_run)
    cfg = json.loads((from_run/"config.json").read_text())
    if weight_lambda is None: weight_lambda = cfg["lamb"]
    img_size = cfg.get("img_size", 224)
    latent_dim = cfg["latent_dim"]

    # load model
    model = ConvSAE(latent_dim=latent_dim).to("cpu")
    state = torch.load(from_run/"model.pt", map_location="cpu")
    model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # build train set = all images from glob EXCLUDING original val paths
    all_paths = sorted(glob(train_glob))
    # reuse original val set for comparability
    val_paths = []
    val_csv = from_run/"paths_val.csv"
    if val_csv.exists():
        import pandas as pd
        val_paths = pd.read_csv(val_csv)["image"].tolist()
    train_paths = [p for p in all_paths if p not in set(val_paths)]
    if not train_paths:
        raise SystemExit("No training images found after excluding val set.")

    train_ds = XRFolder(train_paths, img_size)
    val_ds   = XRFolder(val_paths, img_size) if val_paths else None
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None

    # where to save
    if save_to is None:
        save_to = str(from_run.parent / (from_run.name + "__ft1"))
    out = Path(save_to); out.mkdir(parents=True, exist_ok=True)
    # write config
    new_cfg = dict(cfg); new_cfg.update({
        "parent_run": str(from_run),
        "finetune_epochs": epochs,
        "lr": lr,
        "lamb": weight_lambda,
        "note": "fine-tuned on expanded dataset"
    })
    (out/"config.json").write_text(json.dumps(new_cfg, indent=2))

    # train
    print(f"Train imgs: {len(train_ds)}  |  Val imgs: {len(val_ds) if val_ds else 0}")
    best = None
    for ep in range(1, epochs+1):
        t0 = time.time()
        model.train()
        running = 0.0; cnt = 0
        for x,_ in train_dl:
            x = x.to("cpu")
            xhat, z = model(x)
            rec = F.mse_loss(xhat, x)
            spars = z.abs().mean()
            loss = rec + weight_lambda * spars
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); cnt += 1

        val_mse, mean_active = (math.nan, math.nan)
        if val_dl:
            val_mse, mean_active = eval_val(model, val_dl, weight_lambda)

        print(f"[ep {ep:02d}] loss={running/cnt:.4f}  val_mse={val_mse:.6f}  mean_active≈{mean_active:.2f}  ({time.time()-t0:.1f}s)")

    # save model
    torch.save(model.state_dict(), out/"model.pt")

    # export z_val & paths_val for label mining
    if val_dl:
        zs=[]; paths=[]
        model.eval()
        with torch.no_grad():
            for x,ps in val_dl:
                x = x.to("cpu")
                _, z = model(x)
                zs.append(z.cpu().numpy()); paths += list(ps)
        np.save(out/"z_val.npy", np.concatenate(zs, axis=0))
        import pandas as pd
        pd.DataFrame({"image": paths}).to_csv(out/"paths_val.csv", index=False)
        print("Wrote:", out/"z_val.npy", "and", out/"paths_val.csv")

    print("Saved fine-tuned run to:", out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_run", required=True)
    ap.add_argument("--train_glob", default="data/images_normalized/*.png")
    ap.add_argument("--save_to", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lamb", type=float, default=None)
    args = ap.parse_args()
    train_more(
        from_run=args.from_run,
        train_glob=args.train_glob,
        save_to=args.save_to,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_lambda=args.lamb,
    )