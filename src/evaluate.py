"""Hold-out MAPE, NTL ablation, and Spearman sanity checks. Cell values stay unidentified."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import EPOCHS
from .model import _clone_data, predict_and_scale, prepare_tensors, train_gnn


def _mape(pred: np.ndarray, obs: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.clip(np.abs(obs), 1e-9, None)
    return float(np.mean(np.abs(pred - obs) / denom))


def spatial_holdout_mape(
    df: pd.DataFrame,
    feature_cols: list[str],
    holdout_frac: float = 0.2,
    seed: int = 0,
    epochs: int = EPOCHS,
    output_dir: Path | None = None,
    device: str | None = None,
) -> dict:
    """Train with some districts dropped from L_agg; score pre-scale district sums vs Y_g."""
    districts = np.array(sorted(df["district"].unique()))
    if len(districts) < 5:
        # Delhi reports one NCT-wide Y_g, so there is nothing to hold out.
        return {
            "skipped": True,
            "reason": (
                f"only {len(districts)} reporting unit(s); a spatial hold-out needs at least 5 "
                "so the training split still constrains the loss"
            ),
            "n_districts": int(len(districts)),
        }
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(len(districts) * holdout_frac)))
    hold = set(rng.choice(districts, size=n_hold, replace=False).tolist())
    train_d = [d for d in districts if d not in hold]
    data = prepare_tensors(df, feature_cols)
    mask = torch.tensor([d in set(train_d) for d in data["districts"]], dtype=torch.bool)
    model, history = train_gnn(
        data,
        epochs=epochs,
        seed=seed,
        output_dir=output_dir,
        district_mask=mask,
        tag="holdout",
        device=device,
    )
    payload = _clone_data(data)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    for k in ["x7", "x6", "x5", "e7", "e6", "e5", "e76", "e65", "pop", "d_idx", "y_g"]:
        payload[k] = payload[k].to(device)
    model.eval()
    with torch.no_grad():
        s = model(payload).detach().cpu().numpy()
    tmp = df.copy()
    tmp["z_raw"] = s * tmp["population"].to_numpy()
    sums = tmp.groupby("district").agg(pred=("z_raw", "sum"), obs=("y_inr", "first"))
    hold_df = sums.loc[list(hold)]
    mape = _mape(hold_df["pred"].to_numpy(), hold_df["obs"].to_numpy())
    result = {
        "holdout_districts": sorted(hold),
        "n_train_districts": len(train_d),
        "n_holdout_districts": len(hold),
        "pre_scale_holdout_mape": mape,
        "final_loss": history[-1]["loss"] if history else None,
        "note": "MAPE is on district sums BEFORE dasymetric scale. Post-scale reconstruction ≈ 0 is conservation, not accuracy.",
    }
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "eval_holdout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def ntl_ablation(
    df: pd.DataFrame,
    feature_cols: list[str],
    epochs: int = EPOCHS,
    seed: int = 0,
    output_dir: Path | None = None,
    device: str | None = None,
) -> dict:
    with_ntl = [c for c in feature_cols]
    without_ntl = [c for c in feature_cols if "night_lights" not in c]
    rows = []
    for tag, cols in (("with_ntl", with_ntl), ("without_ntl", without_ntl)):
        if not cols:
            continue
        data = prepare_tensors(df, cols)
        model, history = train_gnn(
            data, epochs=epochs, seed=seed, output_dir=output_dir, tag=f"ablation_{tag}", device=device
        )
        payload = _clone_data(data)
        device_s = device or ("cuda" if torch.cuda.is_available() else "cpu")
        for k in ["x7", "x6", "x5", "e7", "e6", "e5", "e76", "e65", "pop", "d_idx", "y_g"]:
            payload[k] = payload[k].to(device_s)
        scaled = predict_and_scale(model, payload, df)
        eps = 1e-6
        yhat = scaled.groupby("district")["gdp_inr_raw"].sum()
        y = scaled.groupby("district")["y_inr"].first()
        logloss = float(((np.log10(yhat + eps) - np.log10(y + eps)) ** 2).mean())
        rows.append(
            {
                "tag": tag,
                "n_features": len(cols),
                "pre_scale_district_log_loss": logloss,
                "final_train_loss": history[-1]["loss"] if history else None,
            }
        )
        del model
    result = {
        "runs": rows,
        "note": "Ablation compares pre-scale district log-loss with and without night_lights features. NTL is in the model when the raster is present.",
    }
    if output_dir is not None:
        (Path(output_dir) / "eval_ablation_ntl.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def spearman_sanity(df: pd.DataFrame) -> dict:
    """Rank correlations vs lights/roads/POIs — sanity, not ground truth."""
    inhabited = df["population"] > 0
    base = df.loc[inhabited]
    pairs = {
        "wealth_index_vs_night_lights": ("wealth_index", "night_lights"),
        "wealth_index_vs_road_km": ("wealth_index", "road_km"),
        "wealth_index_vs_poi_count": ("wealth_index", "poi_count"),
        "gdp_per_capita_vs_night_lights": ("gdp_per_capita_inr", "night_lights"),
    }
    out = {}
    for name, (a, b) in pairs.items():
        if a not in base.columns or b not in base.columns:
            continue
        rho = base[a].corr(base[b], method="spearman")
        out[name] = None if pd.isna(rho) else float(rho)
    out["note"] = "Spearman vs NTL/roads/POIs is a sanity check, not truth. wealth_index is not a DHS score."
    return out
