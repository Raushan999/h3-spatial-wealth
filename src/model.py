"""Hierarchical GNN for constrained spatial disaggregation (Lee et al. 2026)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import h3

from .config import EPOCHS, GCNII_ALPHA, HIDDEN, LAYERS, LR, OUTPUT, SEEDS


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unique_index(values: pd.Series) -> tuple[list, dict[str, int]]:
    uniq = list(dict.fromkeys(values.tolist()))
    return uniq, {v: i for i, v in enumerate(uniq)}


def build_edges(cells: pd.Series) -> np.ndarray:
    """k=1 H3 adjacency, undirected, no self-loops."""
    present = set(cells)
    src, dst = [], []
    for cell in cells:
        for nb in h3.grid_disk(cell, 1):
            if nb == cell or nb not in present:
                continue
            src.append(cell)
            dst.append(nb)
    mapper = {c: i for i, c in enumerate(cells)}
    e = np.array([[mapper[a], mapper[b]] for a, b in zip(src, dst)], dtype=np.int64)
    return e.T if len(e) else np.zeros((2, 0), dtype=np.int64)


def build_hierarchy(child_cells: list[str], parent_cells: list[str], parent_res: int) -> np.ndarray:
    parent_idx = {p: i for i, p in enumerate(parent_cells)}
    src, dst = [], []
    for i, child in enumerate(child_cells):
        parent = h3.cell_to_parent(child, parent_res)
        if parent in parent_idx:
            src.append(i)
            dst.append(parent_idx[parent])
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


class GraphSageConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nb = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.lin_self(x)
        src, dst = edge_index
        n, d = x.size(0), x.size(1)
        deg = torch.ones(n, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        msg = torch.zeros(n, d, device=x.device)
        msg.index_add_(0, dst, x[src])
        msg = msg / deg.clamp(min=1).unsqueeze(1)
        return F.elu(self.lin_self(x) + self.lin_nb(msg))


class HierarchicalGNN(nn.Module):
    """R7/R6/R5 heterogeneous GNN with adjacency + parent-child message passing."""

    def __init__(self, in_dim: int, hidden: int = HIDDEN, layers: int = LAYERS, alpha: float = GCNII_ALPHA):
        super().__init__()
        self.alpha = alpha
        self.layers = layers
        self.in_r7 = nn.Linear(in_dim, hidden)
        self.in_r6 = nn.Linear(in_dim, hidden)
        self.in_r5 = nn.Linear(in_dim, hidden)
        self.adj7 = nn.ModuleList([GraphSageConv(hidden, hidden) for _ in range(layers)])
        self.adj6 = nn.ModuleList([GraphSageConv(hidden, hidden) for _ in range(layers)])
        self.adj5 = nn.ModuleList([GraphSageConv(hidden, hidden) for _ in range(layers)])
        self.up76 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.up65 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.down67 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.down56 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.mix = nn.Linear(hidden * 2, hidden)
        self.out = nn.Linear(hidden, 1)

    @staticmethod
    def _pool_to_parent(x_child, edge_child_to_parent, n_parent):
        if edge_child_to_parent.numel() == 0:
            return torch.zeros(n_parent, x_child.size(1), device=x_child.device)
        src, dst = edge_child_to_parent
        out = torch.zeros(n_parent, x_child.size(1), device=x_child.device)
        out.index_add_(0, dst, x_child[src])
        deg = torch.zeros(n_parent, device=x_child.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x_child.device))
        return out / deg.clamp(min=1).unsqueeze(1)

    @staticmethod
    def _broadcast_from_parent(x_parent, edge_child_parent, n_child):
        if edge_child_parent.numel() == 0:
            return torch.zeros(n_child, x_parent.size(1), device=x_parent.device)
        child, parent = edge_child_parent
        out = torch.zeros(n_child, x_parent.size(1), device=x_parent.device)
        out[child] = x_parent[parent]
        return out

    def forward(self, data: dict) -> torch.Tensor:
        h7 = F.elu(self.in_r7(data["x7"]))
        h6 = F.elu(self.in_r6(data["x6"]))
        h5 = F.elu(self.in_r5(data["x5"]))
        h7_0, h6_0, h5_0 = h7, h6, h5
        e76, e65 = data["e76"], data["e65"]
        for i in range(self.layers):
            a7 = self.adj7[i](h7, data["e7"])
            a6 = self.adj6[i](h6, data["e6"])
            a5 = self.adj5[i](h5, data["e5"])
            up6 = self._pool_to_parent(h7, e76, h6.size(0))
            up5 = self._pool_to_parent(h6, e65, h5.size(0))
            down6 = self._child_from_parent(h5, e65, h6.size(0), self.down56[i])
            down7 = self._child_from_parent(h6, e76, h7.size(0), self.down67[i])
            h6_new = F.elu(a6 + self.up76[i](up6) + down6)
            h5_new = F.elu(a5 + self.up65[i](up5))
            h7_new = F.elu(self.mix(torch.cat([a7, down7], dim=1)))
            # GCNII-style initial residual
            h7 = (1 - self.alpha) * h7_new + self.alpha * h7_0
            h6 = (1 - self.alpha) * h6_new + self.alpha * h6_0
            h5 = (1 - self.alpha) * h5_new + self.alpha * h5_0
        intensity = F.softplus(self.out(h7)).squeeze(-1)
        return intensity

    def _child_from_parent(self, x_parent, edge_child_parent, n_child, lin):
        if edge_child_parent.numel() == 0:
            return torch.zeros(n_child, x_parent.size(1), device=x_parent.device)
        child, parent = edge_child_parent
        out = torch.zeros(n_child, x_parent.size(1), device=x_parent.device)
        out[child] = lin(x_parent[parent])
        return out


def prepare_tensors(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    r7 = df["h3_r7"].tolist()
    r6 = list(dict.fromkeys(df["h3_r6"].tolist()))
    r5 = list(dict.fromkeys(df["h3_r5"].tolist()))
    x7 = torch.tensor(df[feature_cols].to_numpy(np.float32))
    x6 = torch.tensor(
        df.groupby("h3_r6")[feature_cols].mean().reindex(r6).to_numpy(np.float32)
    )
    x5 = torch.tensor(
        df.groupby("h3_r5")[feature_cols].mean().reindex(r5).to_numpy(np.float32)
    )
    e7 = torch.tensor(build_edges(df["h3_r7"]), dtype=torch.long)
    e6 = torch.tensor(build_edges(pd.Series(r6)), dtype=torch.long)
    e5 = torch.tensor(build_edges(pd.Series(r5)), dtype=torch.long)
    e76 = torch.tensor(build_hierarchy(r7, r6, 6), dtype=torch.long)
    e65 = torch.tensor(build_hierarchy(r6, r5, 5), dtype=torch.long)
    pop = torch.tensor(df["population"].to_numpy(np.float32))
    district_ids, dist_map = _unique_index(df["district"])
    d_idx = torch.tensor(df["district"].map(dist_map).to_numpy(), dtype=torch.long)
    y_g = torch.tensor(
        [df.loc[df["district"] == d, "y_inr"].iloc[0] for d in district_ids],
        dtype=torch.float64,
    )
    return {
        "x7": x7,
        "x6": x6,
        "x5": x5,
        "e7": e7,
        "e6": e6,
        "e5": e5,
        "e76": e76,
        "e65": e65,
        "pop": pop,
        "d_idx": d_idx,
        "y_g": y_g,
        "districts": district_ids,
        "feature_cols": feature_cols,
    }


def _clone_data(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out


def train_gnn(
    data: dict,
    epochs: int = EPOCHS,
    lr: float = LR,
    device: str | None = None,
    seed: int = 0,
    output_dir: Path | None = None,
    district_mask: torch.Tensor | None = None,
    tag: str = "main",
) -> tuple[HierarchicalGNN, list[dict]]:
    """Train one seed. Loss is unchanged: mean((log10 Σ s_i w_i − log10 Y_g)²) + 0.05 * floor."""
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = _clone_data(data)
    model = HierarchicalGNN(in_dim=payload["x7"].shape[1]).to(device)
    for k in ["x7", "x6", "x5", "e7", "e6", "e5", "e76", "e65", "pop", "d_idx", "y_g"]:
        payload[k] = payload[k].to(device)
    if district_mask is not None:
        district_mask = district_mask.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    eps = 1e-6
    tau = 1.0  # INR/person floor (very small in levels; acts as mild regularizer)
    best_state, best_loss = None, float("inf")
    history: list[dict] = []
    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        s = model(payload)
        z = s * payload["pop"]
        yhat = torch.zeros_like(payload["y_g"])
        yhat.index_add_(0, payload["d_idx"], z.double())
        per_d = (torch.log10(yhat + eps) - torch.log10(payload["y_g"] + eps)) ** 2
        if district_mask is None:
            l_agg = per_d.mean()
        else:
            l_agg = per_d[district_mask].mean()
        hinge = torch.clamp(tau - s, min=0) ** 2
        l_floor = (payload["pop"] * hinge).sum() / payload["pop"].sum().clamp(min=1)
        loss = l_agg + 0.05 * l_floor
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        val = float(loss.detach().cpu())
        rec = {
            "epoch": epoch,
            "loss": val,
            "L_agg": float(l_agg.detach().cpu()),
            "L_floor": float(l_floor.detach().cpu()),
            "seed": seed,
            "tag": tag,
        }
        history.append(rec)
        if val < best_loss:
            best_loss = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[{tag} seed={seed}] epoch {epoch:03d}  loss={val:.5f}  L_agg={rec['L_agg']:.5f}")
    if best_state is None:
        raise RuntimeError(
            f"[{tag} seed={seed}] every epoch produced a non-finite loss, so no checkpoint was kept. "
            "This usually means the district mask left no districts in L_agg."
        )
    model.load_state_dict(best_state)
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "loss": best_loss,
        "feature_cols": data["feature_cols"],
        "seed": seed,
        "tag": tag,
        "history": history,
    }
    torch.save(ckpt, out_dir / f"gnn_{tag}_seed{seed}.pt")
    (out_dir / f"loss_{tag}_seed{seed}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model.to(device), history


@torch.no_grad()
def trace_nodes(model: HierarchicalGNN, data: dict, node_idx: list[int], keep: int = 12) -> dict:
    """Replay the forward pass and record what actually flows through one node.

    Used by the walkthrough page so every number shown on screen is a real
    activation from a trained checkpoint rather than an illustration.
    """
    model.eval()
    device = next(model.parameters()).device
    payload = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
    h7 = F.elu(model.in_r7(payload["x7"]))
    h6 = F.elu(model.in_r6(payload["x6"]))
    h5 = F.elu(model.in_r5(payload["x5"]))
    h7_0, h6_0, h5_0 = h7, h6, h5
    e76, e65 = payload["e76"], payload["e65"]

    src, dst = payload["e7"]
    neighbour_of = {i: [] for i in node_idx}
    if src.numel():
        s_np = src.detach().cpu().numpy()
        d_np = dst.detach().cpu().numpy()
        wanted = set(node_idx)
        for a, b in zip(s_np, d_np):
            if int(b) in wanted:
                neighbour_of[int(b)].append(int(a))

    def snap(vec: torch.Tensor) -> dict:
        v = vec.detach().cpu().numpy()
        return {
            "head": [round(float(x), 4) for x in v[:keep]],
            "norm": round(float(np.linalg.norm(v)), 4),
            "mean": round(float(v.mean()), 4),
            "active": int((v > 0).sum()),
            "dim": int(v.shape[0]),
        }

    trace = {int(i): {"input_embedding": snap(h7[i]), "layers": []} for i in node_idx}
    for i in node_idx:
        nbs = neighbour_of[i]
        msg = h7[nbs].mean(dim=0) if nbs else torch.zeros_like(h7[i])
        trace[int(i)]["neighbour_message_layer1"] = snap(msg)
        trace[int(i)]["n_neighbours"] = len(nbs)

    for layer in range(model.layers):
        a7 = model.adj7[layer](h7, payload["e7"])
        a6 = model.adj6[layer](h6, payload["e6"])
        a5 = model.adj5[layer](h5, payload["e5"])
        up6 = model._pool_to_parent(h7, e76, h6.size(0))
        up5 = model._pool_to_parent(h6, e65, h5.size(0))
        down6 = model._child_from_parent(h5, e65, h6.size(0), model.down56[layer])
        down7 = model._child_from_parent(h6, e76, h7.size(0), model.down67[layer])
        h6_new = F.elu(a6 + model.up76[layer](up6) + down6)
        h5_new = F.elu(a5 + model.up65[layer](up5))
        h7_new = F.elu(model.mix(torch.cat([a7, down7], dim=1)))
        for i in node_idx:
            trace[int(i)]["layers"].append(
                {
                    "layer": layer + 1,
                    "neighbour_conv": snap(a7[i]),
                    "from_parent_r6": snap(down7[i]),
                    "mixed": snap(h7_new[i]),
                    "after_initial_residual": snap(
                        (1 - model.alpha) * h7_new[i] + model.alpha * h7_0[i]
                    ),
                    "alpha": model.alpha,
                }
            )
        h7 = (1 - model.alpha) * h7_new + model.alpha * h7_0
        h6 = (1 - model.alpha) * h6_new + model.alpha * h6_0
        h5 = (1 - model.alpha) * h5_new + model.alpha * h5_0

    logit = model.out(h7).squeeze(-1)
    intensity = F.softplus(logit)
    weight = model.out.weight.detach().cpu().numpy().ravel()
    for i in node_idx:
        hv = h7[i].detach().cpu().numpy()
        contrib = hv * weight
        order = np.argsort(-np.abs(contrib))[:keep]
        trace[int(i)]["readout"] = {
            "final_embedding": snap(h7[i]),
            "bias": round(float(model.out.bias.detach().cpu().item()), 5),
            "logit": round(float(logit[i].item()), 5),
            "s_i": round(float(intensity[i].item()), 6),
            "top_units": [
                {"unit": int(u), "activation": round(float(hv[u]), 4),
                 "weight": round(float(weight[u]), 4), "contribution": round(float(contrib[u]), 4)}
                for u in order
            ],
        }
    return {str(k): v for k, v in trace.items()}


def _wealth_from_pci(pci: pd.Series, population: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pci.index, dtype=float)
    inhabited = population > 0
    ranks = pci.loc[inhabited].rank(method="average", pct=True) * 100
    out.loc[inhabited] = ranks
    out.loc[~inhabited] = 0.0
    return out


def predict_and_scale(model: HierarchicalGNN, data: dict, df: pd.DataFrame) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        s = model(data).detach().cpu().numpy()
    out = df.copy()
    out["intensity_raw"] = s
    out["gdp_inr_raw"] = out["intensity_raw"] * out["population"]
    factors = {
        d: g["y_inr"].iloc[0] / max(float(g["gdp_inr_raw"].sum()), 1e-9)
        for d, g in out.groupby("district")
    }
    out["dasymetric_scale"] = out["district"].map(factors).astype(float)
    out["gdp_inr"] = out["gdp_inr_raw"] * out["dasymetric_scale"]
    out["gdp_crore"] = out["gdp_inr"] / 1e7
    out["gdp_per_capita_inr"] = np.where(out["population"] > 0, out["gdp_inr"] / out["population"], np.nan)
    out["wealth_index"] = _wealth_from_pci(out["gdp_per_capita_inr"], out["population"])
    out["wealth_quintile"] = pd.cut(
        out["wealth_index"], bins=[-0.01, 20, 40, 60, 80, 100.01], labels=[1, 2, 3, 4, 5]
    )
    return out


def pre_scale_district_log_loss(intensity: np.ndarray, df: pd.DataFrame) -> float:
    eps = 1e-6
    tmp = df.copy()
    tmp["z"] = intensity * tmp["population"].to_numpy()
    yhat = tmp.groupby("district")["z"].sum()
    y = tmp.groupby("district")["y_inr"].first()
    return float(((np.log10(yhat + eps) - np.log10(y + eps)) ** 2).mean())


def train_ensemble(
    data: dict,
    df: pd.DataFrame,
    epochs: int = EPOCHS,
    seeds: tuple[int, ...] = SEEDS,
    output_dir: Path | None = None,
    device: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """5-seed ensemble: central estimate = mean; also store min–max. Loss/architecture unchanged."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    pci_cols = []
    wi_cols = []
    s_cols = []
    histories = []
    last_out = None
    for seed in seeds:
        payload = _clone_data(data)
        for k in ["x7", "x6", "x5", "e7", "e6", "e5", "e76", "e65", "pop", "d_idx", "y_g"]:
            payload[k] = payload[k].to(device)
        model, history = train_gnn(
            data, epochs=epochs, device=device, seed=seed, output_dir=out_dir, tag="ensemble"
        )
        histories.append(history)
        scaled = predict_and_scale(model, payload, df)
        last_out = scaled
        pci_cols.append(scaled["gdp_per_capita_inr"].to_numpy())
        wi_cols.append(scaled["wealth_index"].to_numpy())
        s_cols.append(scaled["intensity_raw"].to_numpy())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pci = np.vstack(pci_cols)
    wi = np.vstack(wi_cols)
    intens = np.vstack(s_cols)
    out = last_out.copy()
    out["intensity_raw"] = intens.mean(axis=0)
    out["intensity_raw_min"] = intens.min(axis=0)
    out["intensity_raw_max"] = intens.max(axis=0)
    # Re-apply dasymetric scale to the mean intensity so district sums still match Y_g.
    out["gdp_inr_raw"] = out["intensity_raw"] * out["population"]
    factors = {
        d: g["y_inr"].iloc[0] / max(float(g["gdp_inr_raw"].sum()), 1e-9)
        for d, g in out.groupby("district")
    }
    out["dasymetric_scale"] = out["district"].map(factors).astype(float)
    out["gdp_inr"] = out["gdp_inr_raw"] * out["dasymetric_scale"]
    out["gdp_crore"] = out["gdp_inr"] / 1e7
    out["gdp_per_capita_inr"] = np.where(out["population"] > 0, out["gdp_inr"] / out["population"], np.nan)
    out["gdp_per_capita_inr_mean"] = np.nanmean(pci, axis=0)
    out["gdp_per_capita_inr_min"] = np.nanmin(pci, axis=0)
    out["gdp_per_capita_inr_max"] = np.nanmax(pci, axis=0)
    out["wealth_index"] = _wealth_from_pci(out["gdp_per_capita_inr"], out["population"])
    out["wealth_index_mean"] = np.nanmean(wi, axis=0)
    out["wealth_index_min"] = np.nanmin(wi, axis=0)
    out["wealth_index_max"] = np.nanmax(wi, axis=0)
    out["wealth_quintile"] = pd.cut(
        out["wealth_index"], bins=[-0.01, 20, 40, 60, 80, 100.01], labels=[1, 2, 3, 4, 5]
    )
    out["ensemble_caption"] = "ensemble range across seeds, not a calibrated probability of error."
    pre_ll = pre_scale_district_log_loss(out["intensity_raw"].to_numpy(), out)
    (out_dir / "loss_ensemble.json").write_text(
        json.dumps({"seeds": list(seeds), "histories": histories, "pre_scale_district_log_loss": pre_ll}, indent=2),
        encoding="utf-8",
    )
    meta = {
        "n_seeds": int(len(seeds)),
        "seeds": list(seeds),
        "pre_scale_district_log_loss": pre_ll,
        "caption": "ensemble range across seeds, not a calibrated probability of error.",
    }
    return out, meta
