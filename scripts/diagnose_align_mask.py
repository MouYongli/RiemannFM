"""Adapted diagnostic for L_align & L_mask_c on latest run (current API)."""
from __future__ import annotations
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

RUN_DIR = Path(os.environ.get("DIAG_RUN_DIR", "outputs/20260412_235445"))
CKPT_PATH = sorted((RUN_DIR / "checkpoints").glob("step=*.ckpt"))[-1]
cfg = OmegaConf.load(RUN_DIR / ".hydra" / "config.yaml")
print(f"Ckpt: {CKPT_PATH.name}")

from riemannfm.data.datamodule import RiemannFMDataModule
from riemannfm.data.collator import MASK_C, MASK_REAL, MASK_X, RiemannFMGraphCollator
from riemannfm.models.lightning_module import RiemannFMPretrainModule
from torch.utils.data import DataLoader

dm = RiemannFMDataModule(
    data_dir=cfg.data.data_dir, num_edge_types=cfg.data.num_edge_types,
    max_nodes=cfg.data.max_nodes, max_hops=cfg.data.max_hops, num_workers=2,
    text_encoder=cfg.data.text_encoder, val_epoch_size=cfg.data.val_epoch_size,
    batch_size=cfg.training.batch_size,
    mask_ratio_c=float(getattr(cfg.training, "mask_ratio_c", 0.15)),
    mask_ratio_x=float(getattr(cfg.training, "mask_ratio_x", 0.05)),
    rwpe_k=int(getattr(cfg.data, "rwpe_k", 0)),
)
dm.setup("fit")
input_text_dim = dm.dim_text_emb
C_R = dm.relation_text if input_text_dim > 0 else None

module = RiemannFMPretrainModule.from_config(
    model_cfg=cfg.model, manifold_cfg=cfg.manifold, flow_cfg=cfg.flow,
    training_cfg=cfg.training, ablation_cfg=cfg.ablation,
    num_edge_types=cfg.data.num_edge_types, num_entities=cfg.data.num_entities,
    input_text_dim=input_text_dim, C_R=C_R, max_steps=cfg.training.max_steps,
)
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
module.load_state_dict(ckpt["state_dict"], strict=True)
module = module.cuda().eval()
print(f"Loaded, step={CKPT_PATH.stem}")

collator = RiemannFMGraphCollator(
    max_nodes=cfg.data.max_nodes, num_edge_types=cfg.data.num_edge_types,
    mask_ratio_c=float(getattr(cfg.training, "mask_ratio_c", 0.15)),
    mask_ratio_x=float(getattr(cfg.training, "mask_ratio_x", 0.05)),
    rwpe_k=int(getattr(cfg.data, "rwpe_k", 0)),
)
loader = DataLoader(dm._val_dataset, batch_size=cfg.training.batch_size,
                    shuffle=False, num_workers=2, collate_fn=collator)

N_BATCHES = 4
align_stats, mask_stats = [], []
tsweep_a = {0.05:[], 0.07:[], 0.10:[], 0.15:[], 0.20:[]}
tsweep_m = {0.03:[], 0.05:[], 0.07:[], 0.10:[], 0.15:[]}

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for i, batch in enumerate(loader):
        if i >= N_BATCHES: break
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
        E_1 = batch["edge_types"]; node_text = batch["node_text"]
        node_mask = batch["node_mask"]; node_ids = batch["node_ids"]
        mask_type = batch.get("mask_type"); t_node = batch.get("t_node")
        B, N = node_mask.shape

        x_1 = module._get_manifold_coords(node_ids, node_mask)
        true_text_emb = None
        if mask_type is not None and (module.loss_fn.nu_mask_c>0 or module.loss_fn.nu_mask_x>0):
            node_text, true_text_emb = module._apply_mask_semantics(node_text, mask_type)

        with torch.amp.autocast("cuda", enabled=False):
            sample = module.flow.sample(x_1.float(), E_1.float(), node_mask, t_node_override=t_node)

        V_hat, P_hat, h = module.model(
            x_t=sample.x_t, E_t=sample.E_t, t=sample.t,
            node_text=node_text, node_mask=node_mask, C_R=module.C_R,
            node_pe=batch.get("node_pe"),
        )
        h = h.float(); node_text_f = node_text.float()

        # ── L_ALIGN (REAL nodes only) ─────────────────────────────
        real_mask = node_mask & (mask_type == MASK_REAL)
        idx = real_mask.reshape(-1).nonzero(as_tuple=True)[0]
        if idx.numel() > module.loss_fn.max_align_nodes:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:module.loss_fn.max_align_nodes]]
        n_valid = idx.numel()
        if n_valid >= 2:
            h_v = h.reshape(B*N,-1)[idx]; t_v = node_text_f.reshape(B*N,-1)[idx]
            # pre-proj cos
            hn = F.normalize(h_v, dim=-1); tn = F.normalize(t_v, dim=-1)
            M = n_valid; off = ~torch.eye(M, dtype=torch.bool, device=hn.device)
            h_cos = (hn @ hn.T)[off]; t_cos = (tn @ tn.T)[off]
            # projected (with LayerNorm as in training)
            g_proj = module.loss_fn.proj_g(module.loss_fn.ln_g(h_v))
            c_proj = module.loss_fn.proj_c(module.loss_fn.ln_c(t_v))
            gn = F.normalize(g_proj, dim=-1); cn = F.normalize(c_proj, dim=-1)
            g_self = (gn@gn.T)[off]; c_self = (cn@cn.T)[off]
            cross = gn @ cn.T
            diag = cross.diag(); cross_off = cross[off]
            for tau in tsweep_a:
                lg = cross/tau; lab=torch.arange(M, device=lg.device)
                l=(F.cross_entropy(lg,lab)+F.cross_entropy(lg.T,lab))/2
                tsweep_a[tau].append(l.item())
            acc_g2c = (cross.argmax(1)==torch.arange(M,device=cross.device)).float().mean().item()
            acc_c2g = (cross.argmax(0)==torch.arange(M,device=cross.device)).float().mean().item()
            g_svd = torch.linalg.svdvals(gn); c_svd = torch.linalg.svdvals(cn)
            align_stats.append(dict(
                n=M, h_cos=h_cos.mean().item(), t_cos=t_cos.mean().item(),
                g_self=g_self.mean().item(), c_self=c_self.mean().item(),
                diag=diag.mean().item(), off=cross_off.mean().item(),
                gap=(diag.mean()-cross_off.mean()).item(),
                acc_g2c=acc_g2c, acc_c2g=acc_c2g,
                g_norm=g_proj.norm(dim=-1).mean().item(),
                c_norm=c_proj.norm(dim=-1).mean().item(),
                g_sv=g_svd[:5].cpu().tolist(), c_sv=c_svd[:5].cpu().tolist(),
            ))

        # ── L_MASK_C (MASK_C nodes) ───────────────────────────────
        mc_mask = node_mask & (mask_type == MASK_C)
        midx = mc_mask.reshape(-1).nonzero(as_tuple=True)[0]
        n_mc = midx.numel()
        if n_mc >= 2 and true_text_emb is not None:
            h_m = h.reshape(B*N,-1)[midx]
            tgt = true_text_emb.float().reshape(B*N,-1)[midx]
            pred = module.loss_fn.proj_mask_c(h_m)
            pred = pred - pred.mean(0, keepdim=True)
            tgt  = tgt  - tgt.mean(0, keepdim=True)
            pn = F.normalize(pred, dim=-1); tn = F.normalize(tgt, dim=-1)
            hmn = F.normalize(h_m, dim=-1)
            M2 = n_mc; off2 = ~torch.eye(M2, dtype=torch.bool, device=pn.device)
            h_self = (hmn@hmn.T)[off2]; t_self = (tn@tn.T)[off2]
            pt = pn @ tn.T
            diag2 = pt.diag(); off_pt = pt[off2]
            for tau in tsweep_m:
                lg = pt/tau; lab=torch.arange(M2, device=lg.device)
                l=(F.cross_entropy(lg,lab)+F.cross_entropy(lg.T,lab))/2
                tsweep_m[tau].append(l.item())
            acc1 = (pt.argmax(1)==torch.arange(M2,device=pt.device)).float().mean().item()
            acc5 = (pt.topk(min(5,M2),1).indices == torch.arange(M2,device=pt.device).unsqueeze(1)).any(1).float().mean().item()
            ps = (pn@pn.T)[off2]
            mask_stats.append(dict(
                n=M2, h_self=h_self.mean().item(), t_self=t_self.mean().item(),
                diag=diag2.mean().item(), off=off_pt.mean().item(),
                gap=(diag2.mean()-off_pt.mean()).item(),
                acc1=acc1, acc5=acc5, pred_self=ps.mean().item(),
                pred_norm=pred.norm(dim=-1).mean().item(),
            ))
        print(f"  batch {i}: align_n={n_valid} mask_c_n={n_mc}")

def m(s,k): return np.mean([x[k] for x in s])

print("\n== L_ALIGN ==")
if align_stats:
    print(f"  pre-proj  h cos={m(align_stats,'h_cos'):+.4f}  t cos={m(align_stats,'t_cos'):+.4f}")
    print(f"  post-proj g_self={m(align_stats,'g_self'):+.4f}  c_self={m(align_stats,'c_self'):+.4f}")
    print(f"  cross     diag={m(align_stats,'diag'):+.4f}  off={m(align_stats,'off'):+.4f}  gap={m(align_stats,'gap'):+.4f}")
    print(f"  accuracy  g2c={m(align_stats,'acc_g2c'):.4f}  c2g={m(align_stats,'acc_c2g'):.4f}  random={1/m(align_stats,'n'):.4f}")
    print(f"  norms     |g|={m(align_stats,'g_norm'):.3f}  |c|={m(align_stats,'c_norm'):.3f}")
    print(f"  g top5 sv: {[f'{v:.3f}' for v in align_stats[0]['g_sv']]}")
    print(f"  c top5 sv: {[f'{v:.3f}' for v in align_stats[0]['c_sv']]}")
    print("  temp sweep L_align:")
    for tau, ls in sorted(tsweep_a.items()):
        print(f"    τ={tau:.2f}: {np.mean(ls):.4f}")
    print(f"  ln(M) baseline={np.log(m(align_stats,'n')):.4f}")

print("\n== L_MASK_C ==")
if mask_stats:
    print(f"  h_masked self-cos = {m(mask_stats,'h_self'):+.4f}  (collapse if>0.7)")
    print(f"  true_text self-cos= {m(mask_stats,'t_self'):+.4f}")
    print(f"  pred self-cos     = {m(mask_stats,'pred_self'):+.4f}")
    print(f"  pred vs true diag ={m(mask_stats,'diag'):+.4f}  off={m(mask_stats,'off'):+.4f}  gap={m(mask_stats,'gap'):+.4f}")
    print(f"  acc top1={m(mask_stats,'acc1'):.4f}  top5={m(mask_stats,'acc5'):.4f}  random={1/m(mask_stats,'n'):.4f}")
    print(f"  pred norm={m(mask_stats,'pred_norm'):.3f}")
    print("  temp sweep L_mask_c:")
    for tau, ls in sorted(tsweep_m.items()):
        print(f"    τ={tau:.2f}: {np.mean(ls):.4f}")
print("done")
