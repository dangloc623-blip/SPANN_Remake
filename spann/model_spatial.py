"""
SPANN (spatial variant) - lightweight / fast version of model_spatial.py: train in ~5 minutes on a Kaggle GPU.

Same public API as before (``SPANN_model(x_dim, z_dim, enc, dec, class_num, device, tau)``, ``train``, ``train_eval``
with the same arguments and return values); works with the unmodified preprocess.py dataloaders.

Unchanged: loss functions and weights, 3-phase schedule (UOT -> adjacency -> neighbour), hyper-parameter defaults,
  train(): RBF contrastive-margin adjacency loss (sigma, margin, alpha); train_eval(): InfoNCE adjacency loss (tau);
  adaptive-weight spatial neighbour loss + feature neighbour loss.

Changed (execution, same maths): all data + batch sampling on the GPU (no DataLoader iteration / workers), no GPU<->CPU
syncs inside the networks, fused encoder/decoder heads, BCE-with-logits, a single fused Adam, unbalanced Sinkhorn on GPU
(float64) instead of POT on CPU, vectorised losses, cheap dip-test null, evaluation only when needed, no POT / easydl
dependency, lighter default widths (``lite=True``: encoder 512, classifier head 256; ``lite=False`` restores 1024/1024)
and a wall-clock ``time_budget`` (default 240 s) that rescales the three phases so training always ends in time.

Bug fixed: the InfoNCE loss of train_eval() multiplied log_prob (-inf on the diagonal) by a 0/1 mask, i.e. -inf*0 = NaN,
so the loss and then all weights became NaN once the adjacency phase started.
"""
import math
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal, kl_divergence  # noqa: F401  (kept for `from model import *` users)

LOG_EPS = 1e-10


# --------------------------------------------------------------------------------------
# dip test / bimodality (from model_spatial.py; null simulation 10000 -> 300)
# --------------------------------------------------------------------------------------
def calculate_bimodality_coefficient(series=None):
    """
    calculate BC-coefficient, higher BC coef means more likely to be a bimodal distribution
    """
    # Calculate skewness.
    # Correct for statistical sample bias.
    skewness = scipy.stats.skew(
        series,
        axis=None,
        bias=False,
    )
    # Calculate excess kurtosis.
    # Correct for statistical sample bias.
    kurtosis = scipy.stats.kurtosis(
        series,
        axis=None,
        fisher=True,
        bias=False,
    )
    # Calculate count.
    count = len(series)
    # Calculate count factor.
    count_factor = ((count - 1) ** 2) / ((count - 2) * (count - 3))
    # Count bimodality coefficient.
    coefficient = ((skewness**2) + 1) / (kurtosis + (3 * count_factor))
    # Return value.
    return coefficient


def dip(samples, num_bins=100, p=0.99, table=True, n_null=300):
    """
    dip test for distribution, to test is the distribution id multi-modality
    """
    samples = samples / np.abs(samples).max()
    pdf, idxs = np.histogram(samples, bins=num_bins)
    idxs = idxs[:-1] + np.diff(idxs)
    pdf = pdf / pdf.sum()

    cdf = np.cumsum(pdf, dtype=float)
    assert np.abs(cdf[-1] - 1) < 5e-2

    D = 0
    ans = 0
    check = False
    while True:
        gcm_values, gcm_contact_points = gcm_cal(cdf, idxs)
        lcm_values, lcm_contact_points = lcm_cal(cdf, idxs)

        d_gcm, gcm_diff = sup_diff(gcm_values, lcm_values, gcm_contact_points)
        d_lcm, lcm_diff = sup_diff(gcm_values, lcm_values, lcm_contact_points)

        if d_gcm > d_lcm:
            xl = gcm_contact_points[d_gcm == gcm_diff][0]
            xr = lcm_contact_points[lcm_contact_points >= xl][0]
            d = d_gcm
        else:
            xr = lcm_contact_points[d_lcm == lcm_diff][-1]
            xl = gcm_contact_points[gcm_contact_points <= xr][-1]
            d = d_lcm

        gcm_diff_ranged = np.abs(gcm_values[: xl + 1] - cdf[: xl + 1]).max()
        lcm_diff_ranged = np.abs(lcm_values[xr:] - cdf[xr:]).max()

        if d <= D or xr == 0 or xl == cdf.size:
            ans = D
            break
        else:
            D = max(D, gcm_diff_ranged, lcm_diff_ranged)

        cdf = cdf[xl : xr + 1]
        idxs = idxs[xl : xr + 1]
        pdf = pdf[xl : xr + 1]

    if table:
        p_threshold, p_value = p_table(p, ans, samples.size, n_null)
        if ans < p_threshold:
            check = True
        return ans, p_threshold, check, p_value

    return ans


def gcm_cal(cdf, idxs):
    local_cdf = np.copy(cdf)
    local_idxs = np.copy(idxs)
    gcm = [local_cdf[0]]
    contact_points = [0]
    while local_cdf.size > 1:
        distances = local_idxs[1:] - local_idxs[0]
        slopes = (local_cdf[1:] - local_cdf[0]) / distances
        slope_min = slopes.min()
        slope_min_idx = np.where(slopes == slope_min)[0][0] + 1
        gcm.append(local_cdf[0] + distances[:slope_min_idx] * slope_min)
        contact_points.append(contact_points[-1] + slope_min_idx)
        local_cdf = local_cdf[slope_min_idx:]
        local_idxs = local_idxs[slope_min_idx:]
    return np.hstack(gcm), np.hstack(contact_points)


def lcm_cal(cdf, idxs):
    values, points = gcm_cal(1 - cdf[::-1], idxs.max() - idxs[::-1])
    return 1 - values[::-1], idxs.size - points[::-1] - 1


def sup_diff(alpha, beta, contact_points):
    diff = np.abs(alpha[contact_points] - beta[contact_points])
    return diff.max(), diff


def p_table(p, ans, sample_size, n_samples):
    # speed-up: the null distribution only needs a modest number of simulations
    # (original: 10000 x randn(n_cells), which took tens of seconds)
    rng = np.random.default_rng(0)
    dip_sample = [dip(rng.standard_normal(sample_size), table=False) for _ in range(n_samples)]
    dips = np.hstack(dip_sample)
    dip_sample.append(ans)
    index = np.argsort(dip_sample)
    p_value = 1 - (np.argsort(index)[-1] + 1) / len(index)
    return np.percentile(dips, p * 100), p_value


# --------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------
class DomainBN(nn.Module):
    """Domain-specific BatchNorm: one BN per domain, applied to the rows of that domain.
    Rows must be ordered by domain: ``sizes`` = number of rows of domain 0, 1, ... (no numpy, no sync)."""

    def __init__(self, num_features, n_domain=2):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features) for _ in range(n_domain)])

    def forward(self, x, sizes):
        outs = [bn(c) for bn, c in zip(self.bns, torch.split(x, sizes, dim=0))]
        return torch.cat(outs, 0)


class FastEncoder(nn.Module):
    """fc -> BN -> ReLU -> fused [mu | logvar] head.  Source and target go through the same fc in one matmul,
    BN statistics are still computed per domain batch (exactly like the original, which called the encoder
    separately on the source and on the target batch)."""

    def __init__(self, in_dim, hidden, z_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.head = nn.Linear(hidden, 2 * z_dim)
        self.z_dim = z_dim

    def forward(self, x_src, x_tgt=None, sample=True):
        sizes = [len(x_src)] if x_tgt is None else [len(x_src), len(x_tgt)]
        x = x_src if x_tgt is None else torch.cat([x_src, x_tgt], 0)
        h = self.fc(x)
        h = torch.cat([self.bn(c) for c in torch.split(h, sizes, 0)], 0) if len(sizes) > 1 else self.bn(h)
        h = F.relu(h)
        mu, logvar = self.head(h).chunk(2, dim=1)
        var = torch.exp(logvar.clamp(-10.0, 10.0))
        z = mu + var.sqrt() * torch.randn_like(mu) if sample else mu
        return z, mu, var


class FastDecoder(nn.Module):
    """Three one-layer decoders (as in the original). Returns *logits* (sigmoid is folded into the BCE loss).
      common decoder : fc -> domain BN (2 domains)        (source rows use BN #1, target rows BN #0)
      rna decoder    : fc -> BN
      spatial decoder: fc -> BN
    """

    def __init__(self, z_dim, n_common, n_rna, n_spa):
        super().__init__()
        self.fc_c = nn.Linear(z_dim, n_common)
        self.bn_c = DomainBN(n_common, 2)
        self.fc_s = nn.Linear(z_dim, n_rna)
        self.bn_s = nn.BatchNorm1d(n_rna)
        self.fc_t = nn.Linear(z_dim, n_spa)
        self.bn_t = nn.BatchNorm1d(n_spa)

    def forward(self, z_src, z_tgt):
        ns, nt = len(z_src), len(z_tgt)
        hc = self.fc_c(torch.cat([z_tgt, z_src], 0))  # domain 0 (target) first, domain 1 (source) second
        hc = self.bn_c(hc, [nt, ns])
        rec_c_t, rec_c_s = hc[:nt], hc[nt:]
        rec_s = self.bn_s(self.fc_s(z_src))
        rec_t = self.bn_t(self.fc_t(z_tgt))
        return rec_c_s, rec_c_t, rec_s, rec_t


class ProtoCLS(nn.Module):
    def __init__(self, in_dim, out_dim, temp=0.05):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        return self.fc(F.normalize(x)) / self.tmp

    @torch.no_grad()
    def weight_norm(self):
        w = self.fc.weight
        w.div_(w.norm(p=2, dim=1, keepdim=True))


class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mlp=1024, feat_dim=16, temp=0.05):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, hidden_mlp), nn.ReLU(inplace=True), nn.Linear(hidden_mlp, feat_dim)
        )
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        feat = self.projection_head(x)
        return feat, self.ProtoCLS(feat)


# --------------------------------------------------------------------------------------
# losses / OT helpers (all on GPU, no host sync unless stated)
# --------------------------------------------------------------------------------------
def kl_div(mu, var):
    """KL( N(mu, var) || N(0, 1) ), summed over latent dims, mean over batch (closed form)."""
    return (0.5 * (var + mu * mu - 1.0 - torch.log(var))).sum(1).mean()


def distance_gmm(mu_a, mu_b, var_a, var_b):
    """squared 2-Wasserstein-like distance between diagonal gaussians (same as the original)."""
    d_mu = ((mu_a.unsqueeze(1) - mu_b.unsqueeze(0)) ** 2).sum(2)
    d_sd = ((var_a.sqrt().unsqueeze(1) - var_b.sqrt().unsqueeze(0)) ** 2).sum(2)
    return d_mu + d_sd


def pdists(A, squared=False, eps=0):
    prod = A @ A.t()
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()


def mixup_target(x, device=None):
    """all pair-wise mixups x_ij = w_i*x_j + (1-w_i)*x_i, i != j, w~U(0,1)  (Beta(1,1) == Uniform)."""
    b = x.size(0)
    w = torch.rand(b, 1, 1, device=x.device)
    mix = w * x.unsqueeze(0) + (1 - w) * x.unsqueeze(1)  # [i, j, :]
    keep = ~torch.eye(b, dtype=torch.bool, device=x.device)
    return mix[keep]


def sinkhorn_unbalanced(a, b, M, reg=0.01, reg_m=0.5, stop_thr=1e-4, max_iter=1000, check_every=10):
    """Unbalanced Sinkhorn-Knopp, same algorithm / stopping rule as ``ot.unbalanced.sinkhorn_knopp_unbalanced``
    but on GPU. float64 is needed because K = exp(-M/reg) = exp(100*sim) overflows float32."""
    K = torch.exp(-M / reg)
    fi = reg_m / (reg_m + reg)
    u = torch.full_like(a, 1.0 / a.numel())
    v = torch.full_like(b, 1.0 / b.numel())
    for i in range(max_iter):
        u_prev, v_prev = u, v
        u = (a / (K @ v)) ** fi
        v = (b / (K.t() @ u)) ** fi
        if i % check_every == 0:  # one host sync every `check_every` iterations
            err_u = (u - u_prev).abs().max() / torch.stack([u.abs().max(), u_prev.abs().max(), torch.ones_like(u[0])]).max()
            err_v = (v - v_prev).abs().max() / torch.stack([v.abs().max(), v_prev.abs().max(), torch.ones_like(v[0])]).max()
            if bool(0.5 * (err_u + err_v) < stop_thr):
                break
    return u.unsqueeze(1) * K * v.unsqueeze(0)


def ubot_CCD(sim, beta, stop_thr=1e-4):
    """Unbalanced OT between a batch (uniform marginal) and the class prototypes (marginal beta).
    :sim: [n, K] cosine similarities, :beta: [K] float64 tensor
    :return: pseudo label, new beta (float64), normalised plan Q [n,K] (float32), confidence w_i [n]"""
    n = sim.size(0)
    alpha = torch.full((n,), 1.0 / n, dtype=torch.float64, device=sim.device)
    Q = sinkhorn_unbalanced(alpha, beta, -sim.detach().double(), stop_thr=stop_thr)
    Q = Q / Q.sum()
    wt, pseudo = Q.max(1)
    return pseudo, Q.sum(0), Q.float(), wt.float()


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def ensemble_score(logits, confidence_ot):
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    softmax_cls = p.max(1)[0]
    entropy_cls = (p * logp).sum(-1) / np.log(p.size(1)) + 1
    return (entropy_cls + softmax_cls + confidence_ot) / 3, entropy_cls, softmax_cls, confidence_ot


def _to_gpu(x, device, dtype=torch.float32):
    if sp.issparse(x):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x), dtype=dtype).to(device)


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class SPANN_model:
    """
    Same constructor as the original (enc/dec config lists are only used to read the dimensions).
    :x_dim: dict {0: n_common_genes, 1: n_spatial_genes, 2: n_rna_genes}
    :tau: temperature of the InfoNCE adjacency loss used by train_eval
    :lite: lighter networks (encoder hidden 512, classifier head 256); False = original 1024 / 1024
    :enc_hidden / cls_hidden: explicit widths (override `lite`)
    """

    def __init__(self, x_dim, z_dim, enc, dec, class_num, device, tau=0.5, lite=True, enc_hidden=None, cls_hidden=None):
        super().__init__()
        self.tau = tau  # temperature of the InfoNCE adjacency loss (train_eval)
        self.class_num = class_num
        self.device = torch.device(device)
        self.feat_dim = enc[-1][1]
        enc_hidden = enc_hidden or (min(enc[0][1], 512) if lite else enc[0][1])
        cls_hidden = cls_hidden or (256 if lite else 1024)
        self.encoder = FastEncoder(x_dim[0], enc_hidden, z_dim).to(self.device)
        self.decoder = FastDecoder(z_dim, x_dim[0], x_dim[2], x_dim[1]).to(self.device)
        self.classifier = CLS(z_dim, class_num, hidden_mlp=cls_hidden, feat_dim=self.feat_dim).to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------ data
    def _prepare(self, source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor, batch_size=None):
        dev = self.device
        d = {}
        d["xs_c"] = _to_gpu(source_cm_dl.dataset.data, dev)
        d["xt_c"] = _to_gpu(target_cm_dl.dataset.data, dev)
        d["xs_s"] = _to_gpu(source_sp_ds.data, dev)
        d["xt_s"] = _to_gpu(target_sp_ds.data, dev)
        d["ys"] = torch.as_tensor(np.asarray(source_cm_dl.dataset.labels).astype(np.int64)).to(dev)
        coor = torch.as_tensor(np.array(spatial_coor), dtype=torch.float32).to(dev)
        d["coor"] = coor / coor.max(0)[0]
        # class balanced sampling of the source (== WeightedRandomSampler of the original)
        counts = torch.bincount(d["ys"], minlength=self.class_num).float().clamp(min=1)
        d["w_src"] = (1.0 / counts)[d["ys"]]
        d["bs"] = batch_size or getattr(source_cm_dl, "batch_size", None) or 256
        return d

    # ------------------------------------------------------------------ inference pass over all cells
    @torch.no_grad()
    def _infer(self, d, beta, beta_s, resolution, eval_bs=1024, with_source_ot=False, seed=0):
        dev = self.device
        g = torch.Generator(device="cpu").manual_seed(seed)
        proto = self.classifier.ProtoCLS.fc.weight
        ns, nt = len(d["xs_c"]), len(d["xt_c"])
        out = {
            "src_feat": torch.zeros(ns, self.feat_dim, device=dev),
            "tgt_feat": torch.zeros(nt, self.feat_dim, device=dev),
            "escore": torch.zeros(nt, device=dev),
            "entropy": torch.zeros(nt, device=dev),
            "softmax": torch.zeros(nt, device=dev),
            "conf": torch.zeros(nt, device=dev),
            "pred_cls": torch.zeros(nt, dtype=torch.long, device=dev),
            "pred_ot": torch.zeros(nt, dtype=torch.long, device=dev),
            "src_pred_cls": torch.zeros(ns, dtype=torch.long, device=dev),
            "src_pred_ot": torch.zeros(ns, dtype=torch.long, device=dev),
        }
        # random order so that the (per batch) BN statistics are representative; scattered back by index
        perm_s = torch.randperm(ns, generator=g).to(dev)
        perm_t = torch.randperm(nt, generator=g).to(dev)

        for i in range(0, ns, eval_bs):
            idx = perm_s[i : i + eval_bs]
            if len(idx) < 2:
                continue
            z, _, _ = self.encoder(d["xs_c"][idx])
            feat, logit = self.classifier(z)
            out["src_feat"][idx] = feat
            out["src_pred_cls"][idx] = logit.argmax(1)
            if with_source_ot:
                sim = F.normalize(feat) @ proto.t()
                out["src_pred_ot"][idx] = ubot_CCD(sim, beta_s)[0]

        # threshold from mix-up of 256 random target cells (original: the first test batch)
        sub = perm_t[:256]
        z_sub, _, _ = self.encoder(d["xt_c"][sub])
        z_mix = mixup_target(z_sub)
        f_mix, l_mix = self.classifier(z_mix)
        _, _, _, w_mix = ubot_CCD(F.normalize(f_mix) @ proto.t(), beta_s)
        e_mix, _, _, _ = ensemble_score(l_mix, minmax(w_mix))
        thr = torch.minimum(e_mix.mean(), torch.quantile(e_mix, 1 - resolution))

        for i in range(0, nt, eval_bs):
            idx = perm_t[i : i + eval_bs]
            if len(idx) < 2:
                continue
            z, _, _ = self.encoder(d["xt_c"][idx])
            feat, logit = self.classifier(z)
            nf = F.normalize(feat)
            _, _, _, wt = ubot_CCD(nf @ proto.t(), beta)
            e, ent, smx, conf = ensemble_score(logit, minmax(wt))
            pred_cls = logit.argmax(1)
            pred_ot = pred_cls.clone()
            known = e >= thr
            kid = torch.nonzero(known).squeeze(1)
            if kid.numel() > 1:
                pred_ot[kid] = ubot_CCD(nf[kid] @ proto.t(), beta)[0]
            pred_cls[~known] = -1
            pred_ot[~known] = -1
            out["tgt_feat"][idx] = feat
            out["escore"][idx] = e
            out["entropy"][idx] = ent
            out["softmax"][idx] = smx
            out["conf"][idx] = conf
            out["pred_cls"][idx] = pred_cls
            out["pred_ot"][idx] = pred_ot
        out["thr"] = thr
        return {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in out.items()}

    # ------------------------------------------------------------------ core training loop
    def _fit(
        self, d, lr, lambda_recon, lambda_kl, lambda_spa, lambda_cd, lambda_nb, mu, temp, k, resolution,
        novel_cell_test, maxiter, miditer1, miditer2, miditer3, thr_freq, time_budget, log_every, eval_fn=None,
        spa_mode="margin", sigma=0.2, margin=0.5, alpha_neg=0.2, bc_thr=0.555,
    ):
        dev = self.device
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters())
        try:
            opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4, fused=(dev.type == "cuda"))
        except (RuntimeError, TypeError):
            opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)

        bs, ns, nt = d["bs"], len(d["xs_c"]), len(d["xt_c"])
        K = self.class_num
        beta = torch.full((K,), 1.0 / K, dtype=torch.float64, device=dev)
        beta_s = beta.clone()
        threshold = None
        f1, f2, f3 = miditer1 / maxiter, miditer2 / maxiter, miditer3 / maxiter
        t0 = time.time()
        it, dip_done, perm, ptr = 0, False, torch.randperm(nt, device=dev), 0
        history = {}

        def progress(i):
            p = i / maxiter
            return max(p, (time.time() - t0) / time_budget) if time_budget else p

        while progress(it) < 1.0:
            p = progress(it)
            use_uot, use_spa, use_nb = p > f1, p > f3, p > f2

            # ---- batches (GPU sampling)
            si = torch.multinomial(d["w_src"], bs, replacement=True)
            if ptr + bs > nt:
                perm, ptr = torch.randperm(nt, device=dev), 0
            ti = perm[ptr : ptr + bs]
            ptr += bs
            xs_c, xt_c, xs_s, xt_s, ys, coor = d["xs_c"][si], d["xt_c"][ti], d["xs_s"][si], d["xt_s"][ti], d["ys"][si], d["coor"][ti]

            # ---- forward
            z_all, mu_all, var_all = self.encoder(xs_c, xt_c)
            z_s, z_t = z_all[:bs], z_all[bs:]
            mu_s, mu_t, var_s, var_t = mu_all[:bs], mu_all[bs:], var_all[:bs], var_all[bs:]
            rc_s, rc_t, r_s, r_t = self.decoder(z_s, z_t)
            feat_all, logit_all = self.classifier(z_all)
            feat_t, logit_t = feat_all[bs:], logit_all[bs:]
            nfeat_t = F.normalize(feat_t)

            bce = F.binary_cross_entropy_with_logits
            recon = bce(rc_s, xs_c) + bce(rc_t, xt_c) + 0.5 * (bce(r_s, xs_s) + bce(r_t, xt_s))
            kl = kl_div(mu_s, var_s) + kl_div(mu_t, var_t)
            cls_loss = F.cross_entropy(logit_all[:bs], ys, label_smoothing=0.1)

            zero = torch.zeros((), device=dev)
            spa_loss = ccd_loss = neighbor_loss = zero

            if use_uot:
                proto = self.classifier.ProtoCLS.fc.weight
                with torch.no_grad():
                    _, _, _, wti_all = ubot_CCD(nfeat_t @ proto.t(), beta)
                    esc_t, _, _, _ = ensemble_score(logit_t, minmax(wti_all))
                    if threshold is None or it % thr_freq == 0:
                        sub = torch.randperm(bs, device=dev)[:100]
                        z_mix = mixup_target(z_t[sub])
                        f_mix, l_mix = self.classifier(z_mix)
                        _, _, _, w_mix = ubot_CCD(F.normalize(f_mix) @ proto.t(), beta_s)
                        e_mix, _, _, _ = ensemble_score(l_mix, minmax(w_mix))
                        threshold = torch.minimum(e_mix.mean(), torch.quantile(e_mix, 1 - resolution))
                    hc = torch.nonzero(esc_t > threshold).squeeze(1)  # the only host sync of the step
                n_hc = hc.numel()

                if n_hc >= 2:
                    coor_hc = coor[hc]
                    nfeat_hc = nfeat_t[hc]
                    logit_hc = logit_t[hc]
                    with torch.no_grad():
                        _, new_beta, Q_t, wt_i = ubot_CCD(nfeat_hc @ proto.t(), beta)
                        weight_t = minmax(wt_i)
                        soft_q = Q_t / Q_t.sum(1, keepdim=True)
                        ok = (~(torch.isnan(soft_q).any() | torch.isnan(weight_t).any())).float()
                        soft_q = torch.nan_to_num(soft_q)
                        weight_t = torch.nan_to_num(weight_t)
                        beta = torch.where(torch.isnan(new_beta).any(), beta, mu * beta + (1 - mu) * new_beta)

                    if use_spa:
                        cd = pdists(coor_hc, squared=True)
                        if spa_mode == "margin":  # model_spatial.train(): RBF contrastive margin loss
                            feat_sim = nfeat_hc @ nfeat_hc.t()
                            rbf = torch.exp(-cd / (2 * sigma ** 2))
                            knn = torch.topk(cd, k=min(k, n_hc), largest=False)[1]
                            mask = torch.zeros_like(cd).scatter_(1, knn, 1.0)
                            loss_pos = rbf * (1.0 - feat_sim) ** 2
                            loss_neg = (1.0 - rbf) * torch.clamp(feat_sim - margin, min=0.0) ** 2
                            spa_loss = (mask * (loss_pos + alpha_neg * loss_neg)).sum() / (mask.sum() + 1e-10)
                        else:  # model_spatial.train_eval(): InfoNCE on the latent mean
                            z_sp = F.normalize(mu_t[hc], dim=1)
                            eye = torch.eye(n_hc, dtype=torch.bool, device=dev)
                            pos_idx = torch.topk(cd + cd.max() * eye.float(), k=min(k, n_hc - 1), largest=False)[1]
                            pos = torch.zeros_like(eye).scatter_(1, pos_idx, True)
                            sim = (z_sp @ z_sp.t() / self.tau).masked_fill(eye, float("-inf"))
                            log_prob = (sim - torch.logsumexp(sim, 1, keepdim=True)).masked_fill(~pos, 0.0)  # fixes -inf*0=NaN
                            spa_loss = (-log_prob.sum(1) / pos.sum(1).clamp(min=1)).mean()

                    pred_soft = torch.softmax(logit_hc, 1)
                    ccd_loss = ok * torch.mean(weight_t * (-soft_q * torch.log(pred_soft + LOG_EPS)).sum(1))

                    if use_nb:
                        lsm = F.log_softmax(logit_hc, 1)
                        dist = pdists(coor_hc)
                        dist = dist + dist.max() * torch.eye(n_hc, device=dev)
                        sp_idx = dist.argmin(1)
                        a_ij = (nfeat_hc * nfeat_hc[sp_idx]).sum(1).clamp(min=0.0).detach()   # adaptive weight (anti over-smoothing)
                        ce_anchor = -(Q_t[sp_idx] * lsm).sum(1)
                        ce_neighbor = -(Q_t * F.log_softmax(logit_hc[sp_idx], 1)).sum(1)
                        spatial_loss = (a_ij * (ce_anchor + ce_neighbor)).sum() / (2 * n_hc)
                        feat_mat = nfeat_hc @ nfeat_hc.t() / temp
                        feat_mat = feat_mat.masked_fill(torch.eye(n_hc, dtype=torch.bool, device=dev), -1 / temp)
                        lc_idx = feat_mat.argmax(1)
                        local_loss = (
                            -(Q_t[lc_idx] * lsm).sum() - (Q_t * F.log_softmax(logit_hc[lc_idx], 1)).sum()
                        ) / (2 * n_hc)
                        neighbor_loss = (spatial_loss + local_loss) / 2

            loss = (
                lambda_recon * recon + lambda_kl * kl + cls_loss
                + lambda_cd * ccd_loss + lambda_nb * neighbor_loss + lambda_spa * spa_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            self.classifier.ProtoCLS.weight_norm()
            it += 1

            if log_every and it % log_every == 0:
                print(
                    "#Iter %d (%.0fs): Recon %.4f | KL %.3f | CLS %.4f | Spatial %.4f | CCD %.4f | Neighbor %.4f"
                    % (it, time.time() - t0, recon.item(), kl.item(), cls_loss.item(),
                       float(spa_loss), float(ccd_loss), float(neighbor_loss))
                )

            # dip test (decides `resolution`) once, right when the UOT phase is about to start
            if not dip_done and novel_cell_test and progress(it) >= f1:
                dip_done = True
                res = self._infer(d, beta, beta_s, resolution)
                dp = dip(res["escore"], num_bins=100)
                dip_p, dip_ok = dp[-1], dp[-2]
                bc = calculate_bimodality_coefficient(np.sqrt(res["escore"]))
                bimodal = (not dip_ok) or (bc > bc_thr)
                resolution = 0.5 if bimodal else 0.8
                print("[dip test] p=%.4f  BC=%.3f (> %.3f = bimodal)  ood exists: %s  -> resolution=%.1f"
                      % (dip_p, bc, bc_thr, bimodal, resolution))
                if eval_fn is not None:
                    eval_fn(it, res)
                threshold = None

        print("training finished: %d iterations in %.1fs" % (it, time.time() - t0))
        res = self._infer(d, beta, beta_s, resolution, with_source_ot=eval_fn is not None)
        if eval_fn is not None:
            eval_fn(it, res)
        return res

    # ------------------------------------------------------------------ public API
    def train(
        self, source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor,
        test_source_cm_dl, test_target_cm_dl, source_labels, cell_types,
        lr=2e-4, lambda_recon=2000, lambda_kl=0.5, lambda_spa=0.1, lambda_cd=0.001, lambda_nb=0.1,
        mu=0.6, temp=0.1, k=20, resolution=0.5, novel_cell_test=True,
        maxiter=6000, miditer1=2000, miditer2=5000, miditer3=4000, test_freq=1000,
        sigma=0.2, margin=0.5, alpha=0.2, time_budget=240, batch_size=None, thr_freq=None, log_every=None,
    ):
        """Annotate spatial data without ground truth. Arguments as in the original, plus
        :time_budget: wall-clock seconds for the training loop (default 240 -> train() returns in < 5 min); the three phases are
        rescaled to fit, None = run all `maxiter` iterations
        :batch_size: override the dataloader batch size, :thr_freq: threshold refresh period
        :return: adata_source, adata_target, threshold_test"""
        thr_freq = thr_freq or max(1, int(test_freq / 100))   # same refresh period as the original
        log_every = log_every if log_every is not None else max(1, int(test_freq / 10))
        d = self._prepare(source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor, batch_size)
        res = self._fit(d, lr, lambda_recon, lambda_kl, lambda_spa, lambda_cd, lambda_nb, mu, temp, k, resolution,
                        novel_cell_test, maxiter, miditer1, miditer2, miditer3, thr_freq, time_budget, log_every,
                        spa_mode="margin", sigma=sigma, margin=margin, alpha_neg=alpha, bc_thr=0.555)
        source_labels = np.asarray(source_labels).astype(int)
        adata_source = ad.AnnData(sp.csr_matrix(res["src_feat"]))
        adata_source.obs["cell_type"] = [cell_types[i] for i in source_labels]
        adata_source.obs["source"] = "scRNA"
        adata_target = ad.AnnData(sp.csr_matrix(res["tgt_feat"]))
        total = list(cell_types) + ["Unknown"]
        adata_target.obs["pred_cls"] = [total[i] for i in res["pred_cls"]]
        adata_target.obs["pred_ot"] = [total[i] for i in res["pred_ot"]]
        adata_target.obs[["X", "Y"]] = np.array(spatial_coor)
        adata_target.obs["Escore"] = res["escore"]
        adata_target.obs["source"] = "Spatial"
        adata_target.obs.index = adata_target.obs.index.astype(str)
        return adata_source, adata_target, float(res["thr"])

    def train_eval(
        self, source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor,
        test_source_cm_dl, test_target_cm_dl, source_labels, target_labels, cell_types, common_cell_type,
        lr=2e-4, lambda_recon=2000, lambda_kl=0.5, lambda_spa=0.1, lambda_cd=0.001, lambda_nb=0.1,
        mu=0.6, temp=0.1, k=20, resolution=0.5, novel_cell_test=True,
        maxiter=6000, miditer1=2000, miditer2=5000, miditer3=4000, test_freq=1000,
        time_budget=240, batch_size=None, thr_freq=None, log_every=None,
    ):
        """Same as ``train`` but prints accuracies against the spatial ground truth `target_labels`."""
        thr_freq = thr_freq or max(1, int(test_freq / 100))   # same refresh period as the original
        log_every = log_every if log_every is not None else max(1, int(test_freq / 10))
        d = self._prepare(source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor, batch_size)
        source_labels = np.asarray(source_labels).astype(int)
        target_labels = np.asarray(target_labels).astype(int).copy()
        known_mask = np.isin(target_labels, np.unique(source_labels))
        target_eval = target_labels.copy()
        target_eval[~known_mask] = -1

        def report(it, r):
            s_cls = (r["src_pred_cls"] == source_labels).mean()
            s_ot = (r["src_pred_ot"] == source_labels).mean()
            for name in ("ot", "cls"):
                pred = r["pred_" + name]
                tot = (pred == target_eval).mean()
                kn = (pred[known_mask] == target_eval[known_mask]).mean()
                un = (pred[~known_mask] == -1).mean() if (~known_mask).any() else float("nan")
                print("#Iter %d: (%s) Source acc: %f, Target total acc: %f, known acc: %f, unknown acc: %f"
                      % (it, name.upper(), s_ot if name == "ot" else s_cls, tot, kn, un))

        res = self._fit(d, lr, lambda_recon, lambda_kl, lambda_spa, lambda_cd, lambda_nb, mu, temp, k, resolution,
                        novel_cell_test, maxiter, miditer1, miditer2, miditer3, thr_freq, time_budget, log_every,
                        eval_fn=report, spa_mode="infonce", bc_thr=0.45)
        adata_source = ad.AnnData(sp.csr_matrix(res["src_feat"]))
        adata_source.obs["cell_type"] = [cell_types[i] for i in source_labels]
        adata_source.obs["source"] = "scRNA"
        adata_target = ad.AnnData(sp.csr_matrix(res["tgt_feat"]))
        total = list(common_cell_type) + ["Unknown"]
        adata_target.obs["cell_type"] = [cell_types[i] for i in target_labels]
        adata_target.obs["pred_cls"] = [total[i] for i in res["pred_cls"]]
        adata_target.obs["pred_ot"] = [total[i] for i in res["pred_ot"]]
        adata_target.obs[["X", "Y"]] = np.array(spatial_coor)
        adata_target.obs["Escore"] = res["escore"]
        adata_target.obs["source"] = "Spatial"
        adata_target.obs.index = adata_target.obs.index.astype(str)
        return adata_source, adata_target, float(res["thr"])
