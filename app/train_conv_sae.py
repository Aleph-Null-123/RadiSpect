import os, json, argparse, random
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class XRDataset(Dataset):
    def __init__(self, df, img_size=224):
        self.df = df.reset_index(drop=True); self.img_size = img_size
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        p = self.df.loc[i, "image"]
        x = Image.open(p).convert("L").resize((self.img_size, self.img_size), Image.BILINEAR)
        x = np.asarray(x, dtype=np.float32) / 255.0
        return torch.from_numpy(x)[None, ...], p

def split_df(df, train_size, val_size, seed=42):
    rng = np.random.default_rng(seed); idx = np.arange(len(df)); rng.shuffle(idx)
    if train_size == 0: train_size = max(1, len(df) - val_size)
    i_tr = idx[:min(train_size, len(idx))]
    i_va = idx[min(train_size, len(idx)) : min(train_size+val_size, len(idx))]
    return df.iloc[i_tr], df.iloc[i_va]

class ConvSAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,16,3,2,1), nn.ReLU(inplace=True),   # 224->112
            nn.Conv2d(16,32,3,2,1), nn.ReLU(inplace=True),  # 112->56
            nn.Conv2d(32,64,3,2,1), nn.ReLU(inplace=True),  # 56->28
        )
        self.enc_lin = nn.Linear(64*28*28, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 64*28*28)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(inplace=True),  # 28->56
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(inplace=True),  # 56->112
            nn.ConvTranspose2d(16,1, 4,2,1), nn.Sigmoid(),           # 112->224
        )
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return F.relu(self.enc_lin(h))  # non-negative

    def decode(self, z):
        h = self.dec_lin(z).view(-1,64,28,28)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x); xhat = self.decode(z)
        return xhat, z
 
def count_active(z, t=1e-3): return (z>t).float().sum(1).mean().item()

@torch.no_grad()
def evaluate(model, ld, device, t=1e-3):
    model.eval(); mse_sum=0.0; npx=0; act_sum=0.0; n=0
    for x,_ in ld:
        x=x.to(device); xhat,z=model(x)
        mse_sum += F.mse_loss(xhat,x,reduction="sum").item(); npx += x.numel()
        act_sum += count_active(z,t)*x.shape[0]; n += x.shape[0]
    return {"val_mse": (mse_sum/npx) if npx>0 else float("inf"),
            "mean_active": (act_sum/max(1,n))}

def save_json(o,p): p.parent.mkdir(parents=True,exist_ok=True); open(p,"w").write(json.dumps(o,indent=2))
def save_npy(a,p): p.parent.mkdir(parents=True,exist_ok=True); np.save(p,a)

@torch.no_grad()
def export_latents(model, ld, device):
    model.eval(); Z=[]; P=[]
    for x,paths in tqdm(ld, desc="Export latents", leave=False):
        z=model.encode(x.to(device)).cpu().numpy(); Z.append(z); P+=paths
    return (np.concatenate(Z,0) if Z else np.zeros((0,model.enc_lin.out_features))), P

def train_one(cfg, df_pairs, runs_dir, save_epoch="best_mse"):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    torch.set_num_threads(max(1,cfg.get("torch_threads",4)))

    df_tr, df_va = split_df(df_pairs, cfg["train_size"], cfg["val_size"], seed=cfg["seed"])
    tr = XRDataset(df_tr, cfg["img_size"]); va = XRDataset(df_va, cfg["img_size"])
    ld_tr = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    ld_va = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    device=torch.device("cpu"); model=ConvSAE(cfg["latent_dim"]).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    run_name=f"m{cfg['latent_dim']}_lam{cfg['lamb']}_lr{cfg['lr']}_bs{cfg['batch_size']}_e{cfg['epochs']}_seed{cfg['seed']}"
    run_dir=runs_dir/run_name; run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir/"config.json")

    log=[]; best_val=float("inf"); best_state=None; best_epoch=0; best_active=None
    no_improve=0
    for ep in range(1, cfg["epochs"]+1):
        model.train()
        mse_sum=0.0; npx=0
        pbar=tqdm(ld_tr, desc=f"Epoch {ep}/{cfg['epochs']}", leave=False)
        for x,_ in pbar:
            x=x.to(device); xhat,z=model(x)
            mse=F.mse_loss(xhat,x,reduction="mean")
            l1 =z.abs().mean()
            loss=mse+cfg["lamb"]*l1
            opt.zero_grad(); loss.backward(); opt.step()
            mse_sum += mse.item()*x.numel(); npx += x.numel()
            pbar.set_postfix(train_mse=f"{mse.item():.5f}", l1=f"{l1.item():.5f}")

        mets=evaluate(model, ld_va, device, cfg["active_thresh"])
        row={"epoch":ep, "train_mse":(mse_sum/npx) if npx>0 else float("inf"),
             **mets}
        log.append(row)
        pd.DataFrame(log).to_csv(run_dir/"train_log.csv", index=False)

        improved = mets["val_mse"] < best_val
        if improved:
            best_val = mets["val_mse"]; best_epoch = ep; best_active = mets["mean_active"]
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if (ep >= cfg["min_epochs"]) and (no_improve >= cfg["patience"]):
            break

    # choose weights to save
    if save_epoch=="final":
        model_state={k:v.cpu() for k,v in model.state_dict().items()}
        chosen_epoch=log[-1]["epoch"]; chosen_metrics=log[-1]
    else:
        model_state=best_state if best_state is not None else {k:v.cpu() for k,v in model.state_dict().items()}
        chosen_epoch=best_epoch if best_epoch>0 else log[-1]["epoch"]
        chosen_metrics={"val_mse":best_val if best_epoch>0 else log[-1]["val_mse"],
                        "mean_active":best_active if best_epoch>0 else log[-1]["mean_active"]}

    torch.save(model_state, run_dir/"model.pt")
    save_json({"epoch":chosen_epoch, **chosen_metrics}, run_dir/"summary.json")

    model.load_state_dict(model_state)
    Ztr, Ptr = export_latents(model, DataLoader(tr, batch_size=cfg["batch_size"], shuffle=False), device)
    Zva, Pva = export_latents(model, DataLoader(va, batch_size=cfg["batch_size"], shuffle=False), device)
    save_npy(Ztr, run_dir/"z_train.npy"); pd.DataFrame({"image":Ptr}).to_csv(run_dir/"paths_train.csv", index=False)
    save_npy(Zva, run_dir/"z_val.npy");   pd.DataFrame({"image":Pva}).to_csv(run_dir/"paths_val.csv", index=False)
    return run_dir, chosen_epoch, chosen_metrics

def pareto(rows):
    pts = [(i,r["val_mse"], r["mean_active"]) for i,r in enumerate(rows)]
    keep=[]
    for i, vm, ma in pts:
        dominated=False
        for j, vm2, ma2 in pts:
            if j!=i and vm2<=vm and ma2<=ma and (vm2<vm or ma2<ma):
                dominated=True; break
        if not dominated: keep.append(rows[i])
    return keep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs.csv")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--train-size", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=12)             # max epochs
    ap.add_argument("--min-epochs", type=int, default=4)          # early stopping min
    ap.add_argument("--patience", type=int, default=2)            # early stopping patience
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--active-thresh", type=float, default=1e-3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--latents", nargs="+", type=int, default=[128,256])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[1e-3,2e-3])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-3])
    ap.add_argument("--runs-dir", default="models/sae_runs")
    ap.add_argument("--save-epoch", choices=["best_mse","final"], default="best_mse")
    args=ap.parse_args()

    df=pd.read_csv(args.pairs)
    runs_dir=Path(args.runs_dir); runs_dir.mkdir(parents=True, exist_ok=True)

    combos=[]
    for seed in args.seeds:
        for m in args.latents:
            for lam in args.lambdas:
                for lr in args.lrs:
                    combos.append((seed,m,lam,lr))

    leaderboard=[]
    print(f"Planned runs: {len(combos)}")
    for (seed,m,lam,lr) in combos:
        cfg=dict(
            seed=seed, img_size=args.img_size,
            train_size=args.train_size, val_size=args.val_size,
            epochs=args.epochs, min_epochs=args.min_epochs, patience=args.patience,
            batch_size=args.batch_size, active_thresh=args.active_thresh,
            latent_dim=m, lamb=lam, lr=lr,
        )
        run_dir, ep, mets = train_one(cfg, df, runs_dir, save_epoch=args.save_epoch)
        rec={"run":run_dir.name, "epoch":ep, "val_mse":mets["val_mse"], "mean_active":mets["mean_active"]}
        leaderboard.append(rec)
        print(f"[DONE] {run_dir.name}  ep={ep}  val_mse={mets['val_mse']:.6f}  mean_active≈{mets['mean_active']:.2f}")

    lb=pd.DataFrame(leaderboard)
    lb.sort_values("val_mse").to_csv(runs_dir/"leaderboard_by_mse.csv", index=False)
    lb.sort_values(["mean_active","val_mse"]).to_csv(runs_dir/"leaderboard_by_sparsity_then_mse.csv", index=False)
    pd.DataFrame(pareto(leaderboard)).to_csv(runs_dir/"leaderboard_pareto.csv", index=False)

    best_overall = lb.sort_values(["mean_active","val_mse"]).iloc[0].to_dict()
    Path(runs_dir/"best_overall.json").write_text(json.dumps(best_overall, indent=2))
    print("\n== Sparsity-first best (for quick pick) =="); print(json.dumps(best_overall, indent=2))

if __name__=="__main__": main()
