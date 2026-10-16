# network structure
import math
from itertools import cycle

import numpy as np
import ot
import scanpy as sc
import scipy
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from easydl import AccuracyCounter, clear_output, one_hot, variable_to_numpy
from scipy.stats import f, norm, ttest_ind
from torch import nn as nn
from torch.autograd import Function
from torch.distributions import Beta, Normal, kl_divergence
from torch.nn import init
from torch.nn.parameter import Parameter

try:
    from .vade_prior import VaDEPrior
    from .preprocess import generate_ae_params, generate_dataloaders
except (ImportError, ValueError):
    from vade_prior import VaDEPrior
    from preprocess import generate_ae_params, generate_dataloaders


def kl_div(mu, var, weight=None):
    # calculate kl divergence
    loss = kl_divergence(Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))).sum(dim=1)

    # if weight is not None:
    #     loss = loss * weight.squeeze(dim=1)
    return loss.mean()


# def balanced_binary_cross_entropy(recon_x, x):

#     return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    # Returns the matrix of ||x_i-y_j||_p^p via optimized PyTorch CUDA kernel
    if p == 2:
        return torch.cdist(pts_src, pts_dst, p=2.0) ** 2
    return torch.cdist(pts_src, pts_dst, p=float(p)) ** p


def distance_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, var_src: torch.Tensor, var_dst: torch.Tensor):
    # Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances
    std_src = var_src.sqrt()
    std_dst = var_dst.sqrt()
    distance_mean = torch.cdist(mu_src, mu_dst, p=2.0) ** 2
    distance_var = torch.cdist(std_src, std_dst, p=2.0) ** 2
    return distance_mean + distance_var


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


def dip(samples, num_bins=100, p=0.99, table=True):
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
        p_threshold, p_value = p_table(p, ans, samples.size, 10000)
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
    data = [np.random.randn(sample_size) for _ in range(n_samples)]
    dip_sample = [dip(samples, table=False) for samples in data]
    dips = np.hstack(dip_sample)
    dip_sample.append(ans)
    index = np.argsort(dip_sample)
    p_value = 1 - (np.argsort(index)[-1] + 1) / len(index)
    return np.percentile(dips, p * 100), p_value


### Networks
activation = {
    "relu": nn.ReLU(),
    "rrelu": nn.RReLU(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "": None,
}


class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization Layer
    :num_features: dimension of the features
    :n_domain: domain number

    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)  # , requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                # out[indices] = x[indices]
                self.bns[i].training = False
                out[indices] = self.bns[i](x[indices])
                self.bns[i].training = True
        return out


class Block(nn.Module):
    """
    Basic block consist of:
        fc -> bn -> act -> dropout
    :input_dim: dimension of input
    : output_dim: dimension of output
    :norm: batch normalization, '' represent no batch normalization, 1 represent regular batch normalization, int>1 represent domain-specific batch normalization of n domain
    :act: activation function, relu -> nn.ReLU, rrelu -> nn.RReLU, sigmoid -> nn.Sigmoid(), leaky_relu -> nn.LeakyReLU(), tanh -> nn.Tanh(),  '' -> None
    dropout: dropout rate
    """

    def __init__(self, input_dim, output_dim, norm="", act="", dropout=0):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if type(norm) == int:
            if norm == 1:  # TO DO
                self.norm = nn.BatchNorm1d(output_dim)
            else:
                self.norm = DSBatchNorm(output_dim, norm)
        else:
            self.norm = None

        self.act = activation[act]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, y=None):
        h = self.fc(x)
        if self.norm:
            if len(x) == 1:
                pass
            elif self.norm.__class__.__name__ == "DSBatchNorm":
                h = self.norm(h, y)
            else:
                h = self.norm(h)
        if self.act:
            h = self.act(h)
        if self.dropout:
            h = self.dropout(h)
        return h


class NN(nn.Module):
    """
    Neural network consist of multi Blocks
    :input_dim: input dimension
    :cfg: model structure configuration, example - [['fc', x_dim, n_domain, 'sigmoid']]
    """

    def __init__(self, input_dim, cfg):
        super().__init__()
        net = []
        for i, layer in enumerate(cfg):
            if i == 0:
                d_in = input_dim
            if layer[0] == "fc":
                net.append(Block(d_in, *layer[1:]))
            d_in = layer[1]
        self.net = nn.ModuleList(net)

    def forward(self, x, y=None):
        for layer in self.net:
            x = layer(x, y)
        return x


class Encoder(nn.Module):
    """
    VAE Encoder
    :input_dim: input dimension
    :cfg: encoder configuration, example - [['fc', 1024, 1, 'relu'],['fc', 10, '', '']]
    """

    def __init__(self, input_dim, cfg):
        super().__init__()

        enc = []
        mu_enc = []
        var_enc = []

        h_dim = cfg[-2][1]

        enc.append(NN(input_dim[0], cfg[:-1]))
        mu_enc.append(NN(h_dim, cfg[-1:]))
        var_enc.append(NN(h_dim, cfg[-1:]))

        self.enc = nn.ModuleList(enc)
        self.mu_enc = nn.ModuleList(mu_enc)
        self.var_enc = nn.ModuleList(var_enc)

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, domain, y=None):
        """ """
        q = self.enc[domain](x, y)
        mu = self.mu_enc[domain](q, y)
        var = torch.exp(self.var_enc[domain](q, y))
        z = self.reparameterize(mu, var)
        return z, mu, var


class Decoder(nn.Module):
    """
    VAE Decoder
    :z_dim: latent dimension
    :cfg: decoder configuration, example - [['fc', adatas[i].obsm[obsm[i]].shape[1], 1, 'sigmoid']]

    """

    def __init__(self, z_dim, cfg):
        super().__init__()

        dec = []
        for i in cfg.keys():
            dec.append(NN(z_dim, cfg[i]))

        self.dec = nn.ModuleList(dec)

    def forward(self, z, domain, y=None):
        """ """
        reconx_x = self.dec[domain](z, y)

        return reconx_x


class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """

    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """

    def __init__(self, in_dim, out_dim, hidden_mlp=512, feat_dim=16, temp=0.05):
        super(CLS, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, hidden_mlp), nn.ReLU(inplace=True), nn.Linear(hidden_mlp, feat_dim)
        )
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross entropy loss with label smoothing regularizer
    :num_classes (int): number of classes
    :epsilon (float): weight

    """

    def __init__(self, num_classes, device, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)  # 对每一行的所有元素进行softmax运算，并使得每一行所有元素和为1，再取log.

    def forward(self, inputs, targets):
        """
        :inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        :targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=self.device).scatter_(1, targets.unsqueeze(1), 1)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)  # 对行求和，消掉列，keepdim=True则列变成1
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


def sinkhorn_knopp_unbalanced_torch(
    alpha, beta, M, reg=0.01, reg_m=0.5, stopThr=1e-3, max_iter=40, log_v_init=None, check_every=5
):
    """
    Log-domain unbalanced Sinkhorn-Knopp on GPU (float32, no autograd).
    - log-domain: stable with small reg, so no float64 needed (float64 is very slow on consumer GPUs)
    - warm start: `log_v_init` (size = #classes) reuses the scaling vector of the previous call
    - convergence is checked every `check_every` iterations to avoid GPU<->CPU syncs
    :return: transport plan Q (float32), final log_v (for warm start)
    """
    with torch.no_grad():
        device = M.device
        logK = (-M.detach().float()) / reg  # log of the Gibbs kernel
        fi = reg_m / (reg_m + reg)
        log_a = torch.log(torch.as_tensor(alpha, device=device, dtype=torch.float32) + 1e-16)
        log_b = torch.log(torch.as_tensor(beta, device=device, dtype=torch.float32) + 1e-16)
        log_u = torch.full_like(log_a, -math.log(log_a.numel()))
        if log_v_init is not None and log_v_init.shape == log_b.shape:
            log_v = log_v_init.to(device)
        else:
            log_v = torch.full_like(log_b, -math.log(log_b.numel()))

        for it in range(max_iter):
            log_u_prev = log_u
            log_u = fi * (log_a - torch.logsumexp(logK + log_v.unsqueeze(0), dim=1))
            log_v = fi * (log_b - torch.logsumexp(logK + log_u.unsqueeze(1), dim=0))
            if (it + 1) % check_every == 0:
                if torch.max(torch.abs(log_u - log_u_prev)) < stopThr:
                    break

        Q = torch.exp(log_u.unsqueeze(1) + logK + log_v.unsqueeze(0))
    return Q, log_v


# optimal transport alignment
def ubot_CCD(sim, beta, device, stopThr=1e-3, max_iter=40, use_gpu=True, state=None, key=None):
    """
    Unbalanced optimal transport function with GPU acceleration
    :sim: similarity matrix between two distributions
    :beta: marginal distribution of the transport object
    :state/key: optional dict + key to warm-start Sinkhorn from the previous call at the same call-site
    :return: pseudo label computed from the tranport plan, updated marginal distribution of the transport object, normarlized transport plan, confidence score of the transport plan
    """
    if use_gpu and sim.is_cuda:
        try:
            M = -sim
            alpha = torch.ones(sim.size(0), device=device) / sim.size(0)
            log_v_init = state.get(key) if (state is not None and key is not None) else None
            Q_st, log_v = sinkhorn_knopp_unbalanced_torch(
                alpha, beta, M, reg=0.01, reg_m=0.5, stopThr=stopThr, max_iter=max_iter, log_v_init=log_v_init
            )
            if state is not None and key is not None and torch.isfinite(log_v).all():
                state[key] = log_v
            if not torch.isnan(Q_st).any() and not torch.isinf(Q_st).any():
                sum_pi = torch.sum(Q_st)
                Q_st_bar = Q_st / (sum_pi + 1e-12)
                wt_i, pseudo_label = torch.max(Q_st_bar, 1)
                new_beta = torch.sum(Q_st_bar, 0).detach().cpu().numpy()
                return pseudo_label, new_beta, Q_st_bar, wt_i
        except Exception:
            pass

    # CPU POT fallback
    M = -sim
    alpha = ot.unif(sim.size(0))
    Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(
        alpha, beta, M.detach().cpu().numpy(), reg=0.01, reg_m=0.5, stopThr=stopThr, numItermax=max_iter
    )
    Q_st = torch.from_numpy(Q_st).float().to(device)

    # make sum equals to 1
    sum_pi = torch.sum(Q_st)
    Q_st_bar = Q_st / sum_pi
    # confidence score w^t_i
    wt_i, pseudo_label = torch.max(Q_st_bar, 1)

    new_beta = torch.sum(Q_st_bar, 0).cpu().numpy()

    return pseudo_label, new_beta, Q_st_bar, wt_i


def ensemble_score(logit_tensor, confidence_ot):
    """ """
    label_pred_tensor = nn.Softmax(-1)(logit_tensor)
    softmax_cls, _ = torch.max(label_pred_tensor, 1)
    entropy_cls = (label_pred_tensor * torch.log(label_pred_tensor)).sum(-1)
    entropy_cls = entropy_cls / np.log(label_pred_tensor.size()[1]) + 1

    ensemble = (entropy_cls + softmax_cls + confidence_ot) / 3
    return ensemble, entropy_cls, softmax_cls, confidence_ot


def mixup_target(x_batch, device):  ##batchsize*latent
    m = Beta(torch.tensor([1.0], device=device), torch.tensor([1.0], device=device))
    x_batch_expand = x_batch.unsqueeze(1)  ##batchsize * 1 * input_dim
    weight = m.sample([x_batch_expand.size(0), x_batch_expand.size(1)])
    x_batch_mix = weight * x_batch + (1 - weight) * x_batch_expand  ## batchsize * batchsize * input_dim
    x_batch_mix = x_batch_mix.view(-1, x_batch_mix.size(-1))  ## (batchsize * batchsize) * input_dim
    index = torch.arange(0, x_batch_mix.size(0), device=device)  # (batchsize * batchsize)
    x_batch_mix = x_batch_mix[index % (x_batch.size(0) + 1) != 0]  ## (batchsize * (batchsize-1)) * input_dim
    return x_batch_mix


def pdists(A, squared=False, eps=0):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)

    if squared:
        return res
    else:
        res = res.clamp(min=eps).sqrt()
        return res


class SPANN_model:
    """
    SPANN model, using scRNA reference dataset to annotate spatial transcriptome dataset.
    Hỗ trợ 2 cách khởi tạo:
    1. Cách siêu rút gọn (khuyên dùng):
       spann = SPANN_model(adata_cm, adata_spa, adata_rna)
       -> Tự động trích xuất enc, dec, x_dim, z_dim, class_num, device.
    2. Cách cũ (legacy):
       spann = SPANN_model(x_dim, z_dim, enc, dec, class_num, device)
    """

    def __init__(
        self,
        *args,
        adata_cm=None,
        adata_spa=None,
        adata_rna=None,
        class_num=None,
        device=None,
        feat_dim=16,
        hidden_dim=512,
        cls_hidden=512,
        x_dim=None,
        z_dim=None,
        enc=None,
        dec=None,
        **kwargs,
    ):
        super().__init__()
        is_legacy = False
        if len(args) >= 4 and isinstance(args[0], dict):
            is_legacy = True
        elif x_dim is not None and z_dim is not None and enc is not None and dec is not None:
            is_legacy = True

        if is_legacy:
            if len(args) >= 4:
                x_dim, z_dim, enc, dec = args[0], args[1], args[2], args[3]
                if len(args) > 4 and class_num is None:
                    class_num = args[4]
                if len(args) > 5 and device is None:
                    device = args[5]
        else:
            if len(args) >= 3:
                adata_cm, adata_spa, adata_rna = args[0], args[1], args[2]
                if len(args) > 3 and class_num is None:
                    class_num = args[3]
                if len(args) > 4 and device is None:
                    device = args[4]

            if adata_cm is not None and adata_spa is not None and adata_rna is not None:
                self.adata_cm = adata_cm
                self.adata_spa = adata_spa
                self.adata_rna = adata_rna

                if enc is None or dec is None or x_dim is None or z_dim is None:
                    enc, dec, x_dim, z_dim = generate_ae_params(
                        adata_cm, adata_spa, adata_rna, feat_dim=feat_dim
                    )
                    if hidden_dim != 1024 and len(enc) > 1:
                        enc[0][1] = hidden_dim

                if class_num is None:
                    if "cell_type" in adata_rna.obs:
                        class_num = len(np.unique(adata_rna.obs["cell_type"]))
                    elif "labels" in adata_rna.obs:
                        u_lbls = np.unique(adata_rna.obs["labels"])
                        class_num = len(u_lbls[u_lbls >= 0])
                    else:
                        raise ValueError("Không thể tự suy luận class_num, vui lòng truyền class_num!")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if class_num is None or enc is None or dec is None or x_dim is None or z_dim is None:
            raise ValueError(
                "Không đủ thông tin để khởi tạo SPANN_model! "
                "Hãy truyền (adata_cm, adata_spa, adata_rna) hoặc (x_dim, z_dim, enc, dec, class_num, device)."
            )

        self.class_num = class_num
        self.device = device
        self.feat_dim = enc[-1][1]
        self.encoder = Encoder(x_dim, enc).to(device)
        self.decoder = Decoder(z_dim, dec).to(device)
        self.classifier = CLS(z_dim, class_num, hidden_mlp=cls_hidden, feat_dim=self.feat_dim).to(device)
        self.gmm_prior = VaDEPrior(n_components=class_num + 5, latent_dim=z_dim).to(device)

    def _resolve_train_inputs(self, args, kwargs, is_eval=False):
        """
        Tự động xử lý và trích xuất dữ liệu:
        - Cách siêu rút gọn (mới): Không cần truyền tham số nào (nếu đã truyền adata khi khởi tạo)
          hoặc chỉ cần truyền (adata_cm, adata_spa, adata_rna).
        - Cách legacy (cũ): Truyền toàn bộ DataLoader / Dataset / Labels riêng lẻ.
        """
        is_legacy = False
        if len(args) >= 5:
            first_arg = args[0]
            if not (hasattr(first_arg, "obs") and hasattr(first_arg, "X")):
                is_legacy = True

        if is_legacy:
            source_cm_dl = args[0]
            target_cm_dl = args[1]
            source_sp_ds = args[2]
            target_sp_ds = args[3]
            spatial_coor = args[4]
            test_source_cm_dl = args[5] if len(args) > 5 else kwargs.get("test_source_cm_dl", source_cm_dl)
            test_target_cm_dl = args[6] if len(args) > 6 else kwargs.get("test_target_cm_dl", target_cm_dl)
            source_labels = args[7] if len(args) > 7 else kwargs.get("source_labels")
            if is_eval:
                target_labels = args[8] if len(args) > 8 else kwargs.get("target_labels")
                cell_types = args[9] if len(args) > 9 else kwargs.get("cell_types")
                common_cell_type = args[10] if len(args) > 10 else kwargs.get("common_cell_type")
                return (
                    source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor,
                    test_source_cm_dl, test_target_cm_dl, source_labels, target_labels,
                    cell_types, common_cell_type,
                )
            else:
                cell_types = args[8] if len(args) > 8 else kwargs.get("cell_types")
                return (
                    source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor,
                    test_source_cm_dl, test_target_cm_dl, source_labels, cell_types,
                )

        adata_cm = kwargs.get("adata_cm", None)
        adata_spa = kwargs.get("adata_spa", None)
        adata_rna = kwargs.get("adata_rna", None)

        if len(args) >= 3:
            adata_cm, adata_spa, adata_rna = args[0], args[1], args[2]
        elif len(args) == 2:
            adata_cm, adata_spa = args[0], args[1]
        elif len(args) == 1:
            adata_cm = args[0]

        if adata_cm is None:
            adata_cm = getattr(self, "adata_cm", None)
        if adata_spa is None:
            adata_spa = getattr(self, "adata_spa", None)
        if adata_rna is None:
            adata_rna = getattr(self, "adata_rna", None)

        if adata_cm is None or adata_spa is None or adata_rna is None:
            raise ValueError(
                "Thiếu dữ liệu! Vui lòng truyền adata_cm, adata_spa, adata_rna vào hàm khởi tạo SPANN_model "
                "hoặc truyền trực tiếp vào spann.train(adata_cm, adata_spa, adata_rna)."
            )

        batch_size = kwargs.get("batch_size", 256)
        num_workers = kwargs.get("num_workers", 0)

        (
            source_sp_ds,
            target_sp_ds,
            source_cm_dl,
            target_cm_dl,
            test_source_cm_dl,
            test_target_cm_dl,
        ) = generate_dataloaders(
            adata_cm, adata_spa, adata_rna, batch_size=batch_size, num_workers=num_workers
        )

        if "spatial_coor" in kwargs:
            spatial_coor = kwargs["spatial_coor"]
        elif "spatial" in adata_spa.obsm:
            spatial_coor = adata_spa.obsm["spatial"]
        elif "X" in adata_spa.obs and "Y" in adata_spa.obs:
            spatial_coor = adata_spa.obs[["X", "Y"]]
        elif "x" in adata_spa.obs and "y" in adata_spa.obs:
            spatial_coor = adata_spa.obs[["x", "y"]]
        elif "spatial_x" in adata_spa.obs and "spatial_y" in adata_spa.obs:
            spatial_coor = adata_spa.obs[["spatial_x", "spatial_y"]]
        else:
            raise ValueError("Không tìm thấy cột tọa độ ('X', 'Y' hoặc obsm['spatial']) trong adata_spa!")

        if "source_labels" in kwargs:
            source_labels = kwargs["source_labels"]
        elif "labels" in adata_rna.obs:
            source_labels = np.array(adata_rna.obs["labels"])
        else:
            raise ValueError("Không tìm thấy 'labels' trong adata_rna.obs! Vui lòng chạy anndata_preprocess trước.")

        if "cell_types" in kwargs:
            cell_types = kwargs["cell_types"]
        elif "cell_type" in adata_rna.obs:
            cell_types = np.unique(adata_rna.obs["cell_type"])
        else:
            cell_types = np.unique(source_labels)

        if is_eval:
            if "target_labels" in kwargs:
                target_labels = kwargs["target_labels"]
            elif "labels" in adata_spa.obs:
                target_labels = np.array(adata_spa.obs["labels"])
            else:
                raise ValueError("Không tìm thấy 'labels' trong adata_spa.obs cho train_eval!")

            if "common_cell_type" in kwargs:
                common_cell_type = kwargs["common_cell_type"]
            elif "cell_type" in adata_rna.obs and "cell_type" in adata_spa.obs:
                common_cell_type = [
                    c for c in np.unique(adata_rna.obs["cell_type"]) if c in np.unique(adata_spa.obs["cell_type"])
                ]
            else:
                common_cell_type = cell_types

            return (
                source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor,
                test_source_cm_dl, test_target_cm_dl, source_labels, target_labels,
                cell_types, common_cell_type,
            )
        else:
            return (
                source_cm_dl, target_cm_dl, source_sp_ds, target_sp_ds, spatial_coor,
                test_source_cm_dl, test_target_cm_dl, source_labels, cell_types,
            )

    def train(
        self,
        *args,
        lr=2e-4,
        lambda_recon=2000,
        lambda_kl=0.5,
        lambda_spa=0.1,
        lambda_cd=0.001,
        lambda_nb=0.1,
        mu=0.6,
        temp=0.1,
        k=20,
        resolution=0.5,
        novel_cell_test=True,
        maxiter=6000,
        miditer1=2000,
        miditer2=5000,
        miditer3=4000,
        test_freq=1000,
        threshold_freq=200,
        mixup_samples=35,
        sinkhorn_max_iter=40,
        sinkhorn_stop_thr=1e-3,
        use_amp=True,
        **kwargs,
    ):
        """
        Training function for SPANN.
        Hỗ trợ gọi siêu ngắn gọn:
            spann.train()  # Nếu đã truyền adata lúc khởi tạo
            spann.train(adata_cm, adata_spa, adata_rna)
        Hoặc truyền 9 đối số legacy như cũ.
        """
        (
            source_cm_dl,
            target_cm_dl,
            source_sp_ds,
            target_sp_ds,
            spatial_coor,
            test_source_cm_dl,
            test_target_cm_dl,
            source_labels,
            cell_types,
        ) = self._resolve_train_inputs(args, kwargs, is_eval=False)

        optim_enc = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=5e-4)
        optim_dec = torch.optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=5e-4)
        optim_cls = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=5e-4)
        # [MỚI THÊM]
        optim_prior = torch.optim.Adam(self.gmm_prior.parameters(), lr=lr, weight_decay=5e-4)
        
        target_coor = torch.as_tensor(np.array(spatial_coor), dtype=torch.float32, device=self.device)
        target_coor = target_coor / (torch.max(target_coor, dim=0)[0] + 1e-12)

        # Pre-cache specific gene datasets into GPU memory if available for zero CPU-overhead indexing
        try:
            if hasattr(source_sp_ds, "data"):
                raw_s = source_sp_ds.data.toarray() if sp.issparse(source_sp_ds.data) else source_sp_ds.data
                source_sp_gpu = torch.as_tensor(raw_s, dtype=torch.float32, device=self.device)
            else:
                source_sp_gpu = None
        except Exception:
            source_sp_gpu = None

        try:
            if hasattr(target_sp_ds, "data"):
                raw_t = target_sp_ds.data.toarray() if sp.issparse(target_sp_ds.data) else target_sp_ds.data
                target_sp_gpu = torch.as_tensor(raw_t, dtype=torch.float32, device=self.device)
            else:
                target_sp_gpu = None
        except Exception:
            target_sp_gpu = None

        scaler = torch.cuda.amp.GradScaler() if (use_amp and self.device.type == "cuda") else None
        recon_crit = nn.BCELoss()
        cls_criterion = CrossEntropyLabelSmooth(self.class_num, device=self.device, epsilon=0.1).to(self.device)

        beta = None
        beta_s = ot.unif(self.class_num)
        threshold = None
        iteration = 0
        ot_state = {}  # warm-start cache for Sinkhorn scaling vectors

        while iteration < maxiter:
            if len(source_sp_ds) > len(target_sp_ds):
                iters = zip(source_cm_dl, cycle(target_cm_dl))
            else:
                iters = zip(cycle(source_cm_dl), target_cm_dl)
            for minibatch_id, ((im_source, label_source, id_source), (im_target, label_target, id_target)) in enumerate(
                iters
            ):
                label_source = label_source.to(self.device, non_blocking=True)
                label_target = label_target.to(self.device, non_blocking=True)
                im_source = im_source.to(self.device, non_blocking=True)
                im_target = im_target.to(self.device, non_blocking=True)

                if source_sp_gpu is not None:
                    xs_source = source_sp_gpu[id_source]
                else:
                    xs_source = torch.as_tensor(source_sp_ds[id_source.cpu().numpy()][0], dtype=torch.float32, device=self.device)

                if target_sp_gpu is not None:
                    xs_target = target_sp_gpu[id_target]
                else:
                    xs_target = torch.as_tensor(target_sp_ds[id_target.cpu().numpy()][0], dtype=torch.float32, device=self.device)

                y_source = torch.ones(len(im_source), dtype=torch.long, device=self.device)
                y_target = torch.zeros(len(im_target), dtype=torch.long, device=self.device)
                batch_coor = target_coor[id_target]

                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    z_source, mu_source, var_source = self.encoder(im_source, 0, y_source)
                    z_target, mu_target, var_target = self.encoder(im_target, 0, y_target)

                    recon_source_c = self.decoder.dec[0](z_source, y_source)
                    recon_target_c = self.decoder.dec[0](z_target, y_target)
                    recon_source_s = self.decoder.dec[2](z_source, y_source)
                    recon_target_s = self.decoder.dec[1](z_target, y_target)

                    before_lincls_feat_s, after_lincls_s = self.classifier(z_source)
                    before_lincls_feat_t, after_lincls_t = self.classifier(z_target)

                    norm_feat_s = F.normalize(before_lincls_feat_s)
                    norm_feat_t = F.normalize(before_lincls_feat_t)

                    # reconstruction loss
                    recon_loss = (recon_crit(recon_source_c, im_source) + recon_crit(recon_target_c, im_target)) + \
                                 0.5 * (recon_crit(recon_source_s, xs_source) + recon_crit(recon_target_s, xs_target))

                    # KL / GMM loss
                    if iteration < 500:
                        kl_loss = kl_div(mu_source, var_source) + kl_div(mu_target, var_target)
                    else:
                        kl_loss = self.gmm_prior.compute_vade_loss(z_source, mu_source, var_source) + \
                                  self.gmm_prior.compute_vade_loss(z_target, mu_target, var_target)

                    cls_loss = cls_criterion(after_lincls_s, label_source)
                    spa_loss = torch.tensor(0.0, device=self.device)
                    ccd_loss = torch.tensor(0.0, device=self.device)

                if iteration > miditer1:
                    source_prototype = self.classifier.ProtoCLS.fc.weight
                    if beta is None:
                        beta = ot.unif(source_prototype.size()[0])
                    sim_tar_all = torch.matmul(norm_feat_t, source_prototype.t())
                    _, _, Qt_all, wti_all = ubot_CCD(
                        sim_tar_all, beta, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter, state=ot_state, key="all"
                    )
                    confidence_ot_all = (wti_all - torch.min(wti_all)) / (torch.max(wti_all) - torch.min(wti_all) + 1e-12)
                    Escore_t, _, _, _ = ensemble_score(after_lincls_t, confidence_ot_all)
                    ###  calculate threshold
                    if threshold is None or iteration % threshold_freq == 0:
                        with torch.no_grad():
                            idx_rand = torch.randperm(len(z_target), device=self.device)[:mixup_samples]
                            ubot_feature_t_mix = mixup_target(z_target[idx_rand], device=self.device)

                            before_lincls_feat_t_mix, logit_t_mix = self.classifier(ubot_feature_t_mix)
                            norm_feat_t_mix = F.normalize(before_lincls_feat_t_mix)
                            sim_tar_mix = torch.matmul(norm_feat_t_mix, source_prototype.t())
                            _, _, _, wti_mix = ubot_CCD(
                                sim_tar_mix, beta_s, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter, state=ot_state, key="mix"
                            )
                            confidence_ot_mix = (wti_mix - torch.min(wti_mix)) / (
                                torch.max(wti_mix) - torch.min(wti_mix) + 1e-12
                            )

                            Escore_mix_t, _, _, _ = ensemble_score(logit_t_mix, confidence_ot_mix)
                            threshold = torch.mean(Escore_mix_t)
                            threshold = torch.min(
                                torch.tensor([threshold, torch.quantile(Escore_mix_t, (1 - resolution))], device=self.device)
                            )
                    high_conf_idx = Escore_t > threshold
                    high_conf_idx_cpu = high_conf_idx.cpu()
                    id_target_high_conf = id_target[high_conf_idx_cpu]
                    batch_coor_high_conf = batch_coor[high_conf_idx_cpu]

                    # Spatial alignment loss (Vectorized - no python loop)
                    if iteration > miditer3 and len(id_target_high_conf) > 1:
                        spa_dist_mat = distance_gmm(
                            mu_target[high_conf_idx],
                            mu_target[high_conf_idx],
                            var_target[high_conf_idx],
                            var_target[high_conf_idx],
                        )
                        spa_dist_mat = spa_dist_mat / (torch.max(spa_dist_mat) + 1e-12)
                        coor_dist_mat = pdists(batch_coor[high_conf_idx], squared=True)
                        coor_dist_mat = coor_dist_mat / (torch.max(coor_dist_mat) + 1e-12)
                        cur_k = min(k, spa_dist_mat.size(0))
                        if cur_k > 1:
                            index = torch.topk(coor_dist_mat, k=cur_k, largest=False)[1]
                            row_idx = torch.arange(spa_dist_mat.size(0), device=self.device).unsqueeze(1)
                            diff = spa_dist_mat[row_idx, index] - coor_dist_mat[row_idx, index]
                            spa_loss = (torch.norm(diff, dim=1) / (cur_k - 1)).mean()

                    # cell type ot alignment
                    if beta is None:
                        beta = ot.unif(source_prototype.size()[0])
                    sim = torch.matmul(norm_feat_t[high_conf_idx], source_prototype.t())

                    # UOT-based CCD
                    pseudo_label_t_high_conf, new_beta, Q_t, wt_i = ubot_CCD(
                        sim, beta, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter, state=ot_state, key="hc"
                    )
                    pseudo_label_t_soft_high_conf = Q_t * (
                        (1 / torch.sum(Q_t, 1)).unsqueeze(1).expand(-1, Q_t.size()[1])
                    )
                    weight_t = (wt_i - torch.min(wt_i)) / (torch.max(wt_i) - torch.min(wt_i) + 1e-12)
                    if True not in np.isnan(new_beta):
                        beta = mu * beta + (1 - mu) * new_beta
                    pred_label_t = torch.argmax(after_lincls_t, axis=1)
                    pred_label_t_soft = torch.softmax(after_lincls_t, axis=1)

                    if True not in torch.isnan(pseudo_label_t_soft_high_conf):
                        ccd_loss += torch.mean(
                            weight_t
                            * (
                                -pseudo_label_t_soft_high_conf * torch.log(pred_label_t_soft[[high_conf_idx]] + 1e-10)
                            ).sum(dim=1)
                        )

                    # neighbor loss
                    local_loss = torch.tensor(0.0, device=self.device)
                    spatial_loss = torch.tensor(0.0, device=self.device)
                    neighbor_loss = torch.tensor(0.0, device=self.device)
                    if iteration > miditer2 and high_conf_idx.sum() > 1:
                        dist_mat = pdists(batch_coor[high_conf_idx])
                        dist_mat += torch.max(dist_mat) * torch.eye(dist_mat.shape[0], device=self.device)
                        _, spatial_nb_idx = torch.min(dist_mat, 1)
                        spatial_nb_output = after_lincls_t[high_conf_idx][spatial_nb_idx, :]
                        neighbor_Q_spatial = Q_t[spatial_nb_idx, :]
                        spatial_loss += -torch.sum(neighbor_Q_spatial * F.log_softmax(after_lincls_t[high_conf_idx], dim=-1))
                        spatial_loss += -torch.sum(Q_t * F.log_softmax(spatial_nb_output, dim=-1))
                        spatial_loss /= 2 * len(batch_coor_high_conf)

                        feat_mat = torch.matmul(norm_feat_t[high_conf_idx], norm_feat_t[high_conf_idx].t()) / temp
                        mask = torch.eye(feat_mat.size(0), feat_mat.size(0), device=self.device).bool()
                        feat_mat.masked_fill_(mask, -1 / temp)
                        local_nb_dist, local_nb_idx = torch.max(feat_mat, 1)
                        local_nb_output = after_lincls_t[high_conf_idx][local_nb_idx, :]
                        neighbor_Q_local = Q_t[local_nb_idx, :]
                        local_loss += -torch.sum(neighbor_Q_local * F.log_softmax(after_lincls_t[high_conf_idx], dim=-1))
                        local_loss += -torch.sum(Q_t * F.log_softmax(local_nb_output, dim=-1))
                        local_loss /= 2 * len(batch_coor_high_conf)

                        neighbor_loss = (spatial_loss + local_loss) / 2

                    # =====Total Loss=====
                    loss = (
                        lambda_recon * recon_loss
                        + lambda_kl * kl_loss
                        + cls_loss
                        + lambda_cd * ccd_loss
                        + lambda_nb * neighbor_loss
                        + lambda_spa * spa_loss
                    )

                optim_enc.zero_grad(set_to_none=True)
                optim_dec.zero_grad(set_to_none=True)
                optim_cls.zero_grad(set_to_none=True)
                optim_prior.zero_grad(set_to_none=True)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optim_enc)
                    scaler.step(optim_dec)
                    scaler.step(optim_cls)
                    scaler.step(optim_prior)
                    scaler.update()
                else:
                    loss.backward()
                    optim_enc.step()
                    optim_dec.step()
                    optim_cls.step()
                    optim_prior.step()

                self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

                iteration += 1
                if iteration % (test_freq / 10) == 0:
                    print(
                        "#Iter %d: Reconstruction loss: %f, KL loss: %f, CLS loss: %f, Spatial loss: %f, CCD loss: %f, Neighbor loss: %f"
                        % (
                            iteration,
                            recon_loss.item(),
                            kl_loss.item(),
                            cls_loss.item(),
                            spa_loss.item(),
                            ccd_loss.item(),
                            neighbor_loss.item(),
                        )
                    )

                if iteration % test_freq == 0 or iteration == maxiter:
                    with torch.no_grad():
                        source_prototype = self.classifier.ProtoCLS.fc.weight
                        source_z_bank = np.ones([len(source_sp_ds), self.feat_dim])
                        target_z_bank = np.ones([len(target_sp_ds), self.feat_dim])
                        Escore_bank = np.ones(len(target_sp_ds))
                        entropy_cls_bank = np.ones(len(target_sp_ds))
                        softmax_cls_bank = np.ones(len(target_sp_ds))
                        confidence_ot_bank = np.ones(len(target_sp_ds))
                        pred_label_bank_cls = np.ones(len(target_sp_ds))
                        pred_label_bank_ot = np.ones(len(target_sp_ds))
                        threshold_test = None

                        if beta is None:
                            source_prototype = self.classifier.ProtoCLS.fc.weight
                            beta = ot.unif(source_prototype.size()[0])

                        for _, ((im_source, label_source, id_source)) in enumerate(test_source_cm_dl):
                            label_source = label_source.to(self.device, non_blocking=True)
                            im_source = im_source.to(self.device, non_blocking=True)
                            if source_sp_gpu is not None:
                                xs_source = source_sp_gpu[id_source]
                            else:
                                xs_source = torch.as_tensor(source_sp_ds[id_source.cpu().numpy()][0], dtype=torch.float32, device=self.device)
                            y_source = torch.ones(len(im_source), dtype=torch.long, device=self.device)
                            z_source, mu_source, var_source = self.encoder(im_source, 0, y_source)
                            before_lincls_feat_s, after_lincls_s = self.classifier(z_source)
                            source_z_bank[id_source.cpu().numpy(), :] = before_lincls_feat_s.cpu().detach().numpy()
                            pseudo_label_s_cls = torch.argmax(after_lincls_s, axis=1)
                            norm_feat_s = F.normalize(before_lincls_feat_s)
                            sim_src = torch.matmul(norm_feat_s, source_prototype.t())
                            pseudo_label_s_ot, _, _, _ = ubot_CCD(
                                sim_src, beta_s, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter
                            )

                        for _, ((im_target, label_target, id_target)) in enumerate(test_target_cm_dl):
                            im_target = im_target.to(self.device, non_blocking=True)
                            id_target = id_target.to(self.device, non_blocking=True)
                            if target_sp_gpu is not None:
                                xs_target = target_sp_gpu[id_target]
                            else:
                                xs_target = torch.as_tensor(target_sp_ds[id_target.cpu().numpy()][0], dtype=torch.float32, device=self.device)
                            y_target = torch.zeros(len(im_target), dtype=torch.long, device=self.device)
                            z_target, mu_target, var_target = self.encoder(im_target, 0, y_target)
                            before_lincls_feat_t, after_lincls_t = self.classifier(z_target)
                            target_z_bank[id_target.cpu().numpy(), :] = before_lincls_feat_t.cpu().detach().numpy()
                            pseudo_label_t_cls = y_target
                            pseudo_label_t_ot = y_target

                            pseudo_label_t_cls = torch.argmax(after_lincls_t, axis=1)

                            norm_feat_t = F.normalize(before_lincls_feat_t)
                            sim_tar_all = torch.matmul(norm_feat_t, source_prototype.t())
                            _, _, _, wt_i = ubot_CCD(
                                sim_tar_all, beta, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter
                            )
                            confidence_ot_t = (wt_i - torch.min(wt_i)) / (torch.max(wt_i) - torch.min(wt_i) + 1e-12)

                            Escore_t, entropy_cls_t, softmax_cls_t, confidence_ot_t = ensemble_score(
                                after_lincls_t, confidence_ot_t
                            )

                            if threshold_test == None:
                                rand_n = min(mixup_samples, len(z_target))
                                idx_rand_test = torch.randperm(len(z_target), device=self.device)[:rand_n]
                                ubot_feature_t_mix = mixup_target(z_target[idx_rand_test], device=self.device)
                                before_lincls_feat_t_mix, logit_t_mix = self.classifier(ubot_feature_t_mix)
                                norm_feat_t_mix = F.normalize(before_lincls_feat_t_mix)
                                sim_tar_mix = torch.matmul(norm_feat_t_mix, source_prototype.t())
                                _, _, _, wt_i_mix = ubot_CCD(
                                    sim_tar_mix, beta_s, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter
                                )
                                confidence_ot_mix = (wt_i_mix - torch.min(wt_i_mix)) / (
                                    torch.max(wt_i_mix) - torch.min(wt_i_mix) + 1e-12
                                )
                                Escore_mix_t, _, _, _ = ensemble_score(logit_t_mix, confidence_ot_mix)
                                threshold_test = torch.mean(Escore_mix_t)
                                threshold_test = torch.min(
                                    torch.tensor([threshold_test, torch.quantile(Escore_mix_t, (1 - resolution))], device=self.device)
                                )

                            pred_known_id = torch.nonzero(Escore_t >= threshold_test).view(-1)
                            pred_unknown_id = torch.nonzero(Escore_t < threshold_test).view(-1)
                            pseudo_label_t_cls[pred_unknown_id] = -1
                            pseudo_label_t_ot[pred_unknown_id] = -1
                            ##先分known和unknown 对known做ot 避免unknown的样本加进去不符合beta的分布
                            sim_tar_high_conf = torch.matmul(norm_feat_t[pred_known_id], source_prototype.t())
                            if sim_tar_high_conf.size()[0] > 0:
                                pseudo_label_t_high_conf, _, _, _ = ubot_CCD(
                                    sim_tar_high_conf, beta, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter
                                )
                                pseudo_label_t_ot[pred_known_id] = pseudo_label_t_high_conf

                            idx_target = id_target.cpu().numpy()
                            pred_label_bank_cls[idx_target] = pseudo_label_t_cls.cpu().detach().numpy()
                            pred_label_bank_ot[idx_target] = pseudo_label_t_ot.cpu().detach().numpy()
                            Escore_bank[idx_target] = Escore_t.cpu().detach().numpy()
                            entropy_cls_bank[idx_target] = entropy_cls_t.cpu().detach().numpy()
                            softmax_cls_bank[idx_target] = softmax_cls_t.cpu().detach().numpy()
                            confidence_ot_bank[idx_target] = confidence_ot_t.cpu().detach().numpy()

                        if iteration == miditer1 and novel_cell_test:
                            dip_test = dip(Escore_bank, num_bins=100)
                            dip_p_value = dip_test[-1]
                            dip_test = dip_test[-2]
                            ensemble_total_scale = np.sqrt(Escore_bank)
                            BC = calculate_bimodality_coefficient(ensemble_total_scale)
                            print("bimodality of dip test:", dip_p_value, not dip_test)
                            print("bimodality coefficient:(>0.555 indicates bimodality)", BC, BC > 0.555)
                            bimodality = (not dip_test) or (BC > 0.555)
                            print("ood sample exists:", bimodality)
                            if bimodality:
                                resolution = 0.5
                            else:
                                resolution = 0.8

        adata_source = sc.AnnData(sp.csr_matrix(source_z_bank))
        adata_source.obs["cell_type"] = [cell_types[i] for i in source_labels.astype(int)]
        adata_source.obs["source"] = "scRNA"

        adata_target = sc.AnnData(sp.csr_matrix(target_z_bank))
        total_cell_types = list(cell_types) + ["Unknown"]
        adata_target.obs["pred_cls"] = [total_cell_types[i] for i in pred_label_bank_cls.astype(int)]
        adata_target.obs["pred_ot"] = [total_cell_types[i] for i in pred_label_bank_ot.astype(int)]
        adata_target.obs[["X", "Y"]] = np.array(spatial_coor)
        adata_target.obs["Escore"] = Escore_bank
        adata_target.obs["source"] = "Spatial"

        return adata_source, adata_target, threshold_test

    def train_eval(
        self,
        *args,
        lr=2e-4,
        lambda_recon=2000,
        lambda_kl=0.5,
        lambda_spa=0.1,
        lambda_cd=0.001,
        lambda_nb=0.1,
        mu=0.6,
        temp=0.1,
        k=20,
        resolution=0.5,
        novel_cell_test=True,
        maxiter=6000,
        miditer1=2000,
        miditer2=5000,
        miditer3=4000,
        test_freq=1000,
        threshold_freq=200,
        mixup_samples=35,
        sinkhorn_max_iter=40,
        sinkhorn_stop_thr=1e-3,
        use_amp=True,
        **kwargs,
    ):
        """
        Training and evaluating function for SPANN.
        Hỗ trợ gọi siêu ngắn gọn:
            spann.train_eval()
            spann.train_eval(adata_cm, adata_spa, adata_rna)
        Hoặc truyền 11 đối số legacy như cũ.
        """
        (
            source_cm_dl,
            target_cm_dl,
            source_sp_ds,
            target_sp_ds,
            spatial_coor,
            test_source_cm_dl,
            test_target_cm_dl,
            source_labels,
            target_labels,
            cell_types,
            common_cell_type,
        ) = self._resolve_train_inputs(args, kwargs, is_eval=True)
        optim_enc = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=5e-4)
        optim_dec = torch.optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=5e-4)
        optim_cls = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=5e-4)
        optim_prior = torch.optim.Adam(self.gmm_prior.parameters(), lr=lr, weight_decay=5e-4)
        
        target_coor = torch.as_tensor(np.array(spatial_coor), dtype=torch.float32, device=self.device)
        target_coor = target_coor / (torch.max(target_coor, dim=0)[0] + 1e-12)

        # Pre-cache specific gene datasets into GPU memory if available for zero CPU-overhead indexing
        try:
            if hasattr(source_sp_ds, "data"):
                raw_s = source_sp_ds.data.toarray() if sp.issparse(source_sp_ds.data) else source_sp_ds.data
                source_sp_gpu = torch.as_tensor(raw_s, dtype=torch.float32, device=self.device)
            else:
                source_sp_gpu = None
        except Exception:
            source_sp_gpu = None

        try:
            if hasattr(target_sp_ds, "data"):
                raw_t = target_sp_ds.data.toarray() if sp.issparse(target_sp_ds.data) else target_sp_ds.data
                target_sp_gpu = torch.as_tensor(raw_t, dtype=torch.float32, device=self.device)
            else:
                target_sp_gpu = None
        except Exception:
            target_sp_gpu = None

        scaler = torch.cuda.amp.GradScaler() if (use_amp and self.device.type == "cuda") else None
        recon_crit = nn.BCELoss()
        cls_criterion = CrossEntropyLabelSmooth(self.class_num, device=self.device, epsilon=0.1).to(self.device)

        beta = None
        beta_s = ot.unif(self.class_num)
        threshold = None
        iteration = 0
        ot_state = {}  # warm-start cache for Sinkhorn scaling vectors

        while iteration < maxiter:
            if len(source_sp_ds) > len(target_sp_ds):
                iters = zip(source_cm_dl, cycle(target_cm_dl))
            else:
                iters = zip(cycle(source_cm_dl), target_cm_dl)
            for minibatch_id, ((im_source, label_source, id_source), (im_target, label_target, id_target)) in enumerate(
                iters
            ):
                label_source = label_source.to(self.device, non_blocking=True)
                label_target = label_target.to(self.device, non_blocking=True)
                im_source = im_source.to(self.device, non_blocking=True)
                im_target = im_target.to(self.device, non_blocking=True)

                if source_sp_gpu is not None:
                    xs_source = source_sp_gpu[id_source]
                else:
                    xs_source = torch.as_tensor(source_sp_ds[id_source.cpu().numpy()][0], dtype=torch.float32, device=self.device)

                if target_sp_gpu is not None:
                    xs_target = target_sp_gpu[id_target]
                else:
                    xs_target = torch.as_tensor(target_sp_ds[id_target.cpu().numpy()][0], dtype=torch.float32, device=self.device)

                y_source = torch.ones(len(im_source), dtype=torch.long, device=self.device)
                y_target = torch.zeros(len(im_target), dtype=torch.long, device=self.device)
                batch_coor = target_coor[id_target]

                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    z_source, mu_source, var_source = self.encoder(im_source, 0, y_source)
                    z_target, mu_target, var_target = self.encoder(im_target, 0, y_target)

                    recon_source_c = self.decoder.dec[0](z_source, y_source)
                    recon_target_c = self.decoder.dec[0](z_target, y_target)
                    recon_source_s = self.decoder.dec[2](z_source, y_source)
                    recon_target_s = self.decoder.dec[1](z_target, y_target)

                    before_lincls_feat_s, after_lincls_s = self.classifier(z_source)
                    before_lincls_feat_t, after_lincls_t = self.classifier(z_target)

                    norm_feat_s = F.normalize(before_lincls_feat_s)
                    norm_feat_t = F.normalize(before_lincls_feat_t)

                    # reconstruction loss
                    recon_loss = (recon_crit(recon_source_c, im_source) + recon_crit(recon_target_c, im_target)) + \
                                 0.5 * (recon_crit(recon_source_s, xs_source) + recon_crit(recon_target_s, xs_target))

                    # KL / GMM loss
                    if iteration < 500:
                        kl_loss = kl_div(mu_source, var_source) + kl_div(mu_target, var_target)
                    else:
                        kl_loss = self.gmm_prior.compute_vade_loss(z_source, mu_source, var_source) + \
                                  self.gmm_prior.compute_vade_loss(z_target, mu_target, var_target)

                    cls_loss = cls_criterion(after_lincls_s, label_source)
                    spa_loss = torch.tensor(0.0, device=self.device)
                    ccd_loss = torch.tensor(0.0, device=self.device)

                if iteration > miditer1:
                    source_prototype = self.classifier.ProtoCLS.fc.weight
                    if beta is None:
                        beta = ot.unif(source_prototype.size()[0])
                    sim_tar_all = torch.matmul(norm_feat_t, source_prototype.t())
                    _, _, Qt_all, wti_all = ubot_CCD(
                        sim_tar_all, beta, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter, state=ot_state, key="all"
                    )
                    confidence_ot_all = (wti_all - torch.min(wti_all)) / (torch.max(wti_all) - torch.min(wti_all) + 1e-12)
                    Escore_t, _, _, _ = ensemble_score(after_lincls_t, confidence_ot_all)
                    ###  calculate threshold
                    if threshold is None or iteration % threshold_freq == 0:
                        with torch.no_grad():
                            idx_rand = torch.randperm(len(z_target), device=self.device)[:mixup_samples]
                            ubot_feature_t_mix = mixup_target(z_target[idx_rand], device=self.device)

                            before_lincls_feat_t_mix, logit_t_mix = self.classifier(ubot_feature_t_mix)
                            norm_feat_t_mix = F.normalize(before_lincls_feat_t_mix)
                            sim_tar_mix = torch.matmul(norm_feat_t_mix, source_prototype.t())
                            _, _, _, wti_mix = ubot_CCD(
                                sim_tar_mix, beta_s, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter, state=ot_state, key="mix"
                            )
                            confidence_ot_mix = (wti_mix - torch.min(wti_mix)) / (
                                torch.max(wti_mix) - torch.min(wti_mix) + 1e-12
                            )

                            Escore_mix_t, _, _, _ = ensemble_score(logit_t_mix, confidence_ot_mix)
                            threshold = torch.mean(Escore_mix_t)
                            threshold = torch.min(
                                torch.tensor([threshold, torch.quantile(Escore_mix_t, (1 - resolution))], device=self.device)
                            )
                    high_conf_idx = Escore_t > threshold
                    high_conf_idx_cpu = high_conf_idx.cpu()
                    id_target_high_conf = id_target[high_conf_idx_cpu]
                    batch_coor_high_conf = batch_coor[high_conf_idx_cpu]

                    # Spatial alignment loss (Vectorized - no python loop)
                    if iteration > miditer3 and len(id_target_high_conf) > 1:
                        spa_dist_mat = distance_gmm(
                            mu_target[high_conf_idx],
                            mu_target[high_conf_idx],
                            var_target[high_conf_idx],
                            var_target[high_conf_idx],
                        )
                        spa_dist_mat = spa_dist_mat / (torch.max(spa_dist_mat) + 1e-12)
                        coor_dist_mat = pdists(batch_coor[high_conf_idx], squared=True)
                        coor_dist_mat = coor_dist_mat / (torch.max(coor_dist_mat) + 1e-12)
                        cur_k = min(k, spa_dist_mat.size(0))
                        if cur_k > 1:
                            index = torch.topk(coor_dist_mat, k=cur_k, largest=False)[1]
                            row_idx = torch.arange(spa_dist_mat.size(0), device=self.device).unsqueeze(1)
                            diff = spa_dist_mat[row_idx, index] - coor_dist_mat[row_idx, index]
                            spa_loss = (torch.norm(diff, dim=1) / (cur_k - 1)).mean()

                    # cell type ot alignment
                    if beta is None:
                        beta = ot.unif(source_prototype.size()[0])
                    sim = torch.matmul(norm_feat_t[high_conf_idx], source_prototype.t())

                    # UOT-based CCD
                    pseudo_label_t_high_conf, new_beta, Q_t, wt_i = ubot_CCD(
                        sim, beta, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter, state=ot_state, key="hc"
                    )
                    pseudo_label_t_soft_high_conf = Q_t * (
                        (1 / torch.sum(Q_t, 1)).unsqueeze(1).expand(-1, Q_t.size()[1])
                    )
                    weight_t = (wt_i - torch.min(wt_i)) / (torch.max(wt_i) - torch.min(wt_i) + 1e-12)
                    if True not in np.isnan(new_beta):
                        beta = mu * beta + (1 - mu) * new_beta
                    pred_label_t = torch.argmax(after_lincls_t, axis=1)
                    pred_label_t_soft = torch.softmax(after_lincls_t, axis=1)

                    if True not in torch.isnan(pseudo_label_t_soft_high_conf):
                        ccd_loss += torch.mean(
                            weight_t
                            * (
                                -pseudo_label_t_soft_high_conf * torch.log(pred_label_t_soft[[high_conf_idx]] + 1e-10)
                            ).sum(dim=1)
                        )

                    # neighbor loss
                    local_loss = torch.tensor(0.0, device=self.device)
                    spatial_loss = torch.tensor(0.0, device=self.device)
                    neighbor_loss = torch.tensor(0.0, device=self.device)
                    if iteration > miditer2 and high_conf_idx.sum() > 1:
                        dist_mat = pdists(batch_coor[high_conf_idx])
                        dist_mat += torch.max(dist_mat) * torch.eye(dist_mat.shape[0], device=self.device)
                        _, spatial_nb_idx = torch.min(dist_mat, 1)
                        spatial_nb_output = after_lincls_t[high_conf_idx][spatial_nb_idx, :]
                        neighbor_Q_spatial = Q_t[spatial_nb_idx, :]
                        spatial_loss += -torch.sum(neighbor_Q_spatial * F.log_softmax(after_lincls_t[high_conf_idx], dim=-1))
                        spatial_loss += -torch.sum(Q_t * F.log_softmax(spatial_nb_output, dim=-1))
                        spatial_loss /= 2 * len(batch_coor_high_conf)

                        feat_mat = torch.matmul(norm_feat_t[high_conf_idx], norm_feat_t[high_conf_idx].t()) / temp
                        mask = torch.eye(feat_mat.size(0), feat_mat.size(0), device=self.device).bool()
                        feat_mat.masked_fill_(mask, -1 / temp)
                        local_nb_dist, local_nb_idx = torch.max(feat_mat, 1)
                        local_nb_output = after_lincls_t[high_conf_idx][local_nb_idx, :]
                        neighbor_Q_local = Q_t[local_nb_idx, :]
                        local_loss += -torch.sum(neighbor_Q_local * F.log_softmax(after_lincls_t[high_conf_idx], dim=-1))
                        local_loss += -torch.sum(Q_t * F.log_softmax(local_nb_output, dim=-1))
                        local_loss /= 2 * len(batch_coor_high_conf)

                        neighbor_loss = (spatial_loss + local_loss) / 2

                    # =====Total Loss=====
                    loss = (
                        lambda_recon * recon_loss
                        + lambda_kl * kl_loss
                        + cls_loss
                        + lambda_cd * ccd_loss
                        + lambda_nb * neighbor_loss
                        + lambda_spa * spa_loss
                    )

                optim_enc.zero_grad(set_to_none=True)
                optim_dec.zero_grad(set_to_none=True)
                optim_cls.zero_grad(set_to_none=True)
                optim_prior.zero_grad(set_to_none=True)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optim_enc)
                    scaler.step(optim_dec)
                    scaler.step(optim_cls)
                    scaler.step(optim_prior)
                    scaler.update()
                else:
                    loss.backward()
                    optim_enc.step()
                    optim_dec.step()
                    optim_cls.step()
                    optim_prior.step()
                self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

                iteration += 1
                if iteration % (test_freq / 10) == 0:
                    print(
                        "#Iter %d: Reconstruction loss: %f, KL loss: %f, CLS loss: %f, Spatial loss: %f, CCD loss: %f, Neighbor loss: %f"
                        % (
                            iteration,
                            recon_loss.item(),
                            kl_loss.item(),
                            cls_loss.item(),
                            spa_loss.item(),
                            ccd_loss.item(),
                            neighbor_loss.item(),
                        )
                    )

                if iteration % test_freq == 0 or iteration == maxiter:
                    with torch.no_grad():
                        source_corrcnts_ot = 0
                        source_corrcnts_cls = 0
                        source_cnts = 0
                        target_known_cnts = 0
                        target_known_corrcnts_ot = 0
                        target_known_corrcnts_cls = 0
                        target_unknown_cnts = 0
                        target_unknown_corrcnts_ot = 0
                        target_unknown_corrcnts_cls = 0
                        source_prototype = self.classifier.ProtoCLS.fc.weight
                        source_z_bank = np.ones([len(source_sp_ds), self.feat_dim])
                        target_z_bank = np.ones([len(target_sp_ds), self.feat_dim])
                        Escore_bank = np.ones(len(target_sp_ds))
                        entropy_cls_bank = np.ones(len(target_sp_ds))
                        softmax_cls_bank = np.ones(len(target_sp_ds))
                        confidence_ot_bank = np.ones(len(target_sp_ds))
                        pred_label_bank_cls = np.ones(len(target_sp_ds))
                        pred_label_bank_ot = np.ones(len(target_sp_ds))
                        threshold_test = None

                        if beta is None:
                            source_prototype = self.classifier.ProtoCLS.fc.weight
                            beta = ot.unif(source_prototype.size()[0])

                        for _, ((im_source, label_source, id_source)) in enumerate(test_source_cm_dl):
                            label_source = label_source.to(self.device, non_blocking=True)
                            im_source = im_source.to(self.device, non_blocking=True)
                            if source_sp_gpu is not None:
                                xs_source = source_sp_gpu[id_source]
                            else:
                                xs_source = torch.as_tensor(source_sp_ds[id_source.cpu().numpy()][0], dtype=torch.float32, device=self.device)
                            y_source = torch.ones(len(im_source), dtype=torch.long, device=self.device)
                            z_source, mu_source, var_source = self.encoder(im_source, 0, y_source)
                            before_lincls_feat_s, after_lincls_s = self.classifier(z_source)
                            source_z_bank[id_source.cpu().numpy(), :] = before_lincls_feat_s.cpu().detach().numpy()
                            pseudo_label_s_cls = torch.argmax(after_lincls_s, axis=1)
                            norm_feat_s = F.normalize(before_lincls_feat_s)
                            sim_src = torch.matmul(norm_feat_s, source_prototype.t())
                            pseudo_label_s_ot, _, _, _ = ubot_CCD(
                                sim_src, beta_s, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter
                            )

                            source_corrcnts_ot += torch.sum(pseudo_label_s_ot == label_source)
                            source_corrcnts_cls += torch.sum(pseudo_label_s_cls == label_source)
                            source_cnts += len(label_source)
                        for _, ((im_target, label_target, id_target)) in enumerate(test_target_cm_dl):
                            index_tf_known = [
                                label_target[i] in list(np.unique(source_labels)) for i in range(len(label_target))
                            ]
                            index_known = [i for i in range(len(index_tf_known)) if index_tf_known[i]]
                            index_unknown = [i for i in range(len(index_tf_known)) if not index_tf_known[i]]

                            im_target = im_target.to(self.device, non_blocking=True)
                            id_target = id_target.to(self.device, non_blocking=True)
                            if target_sp_gpu is not None:
                                xs_target = target_sp_gpu[id_target]
                            else:
                                xs_target = torch.as_tensor(target_sp_ds[id_target.cpu().numpy()][0], dtype=torch.float32, device=self.device)
                            y_target = torch.zeros(len(im_target), dtype=torch.long, device=self.device)
                            z_target, mu_target, var_target = self.encoder(im_target, 0, y_target)
                            before_lincls_feat_t, after_lincls_t = self.classifier(z_target)
                            target_z_bank[id_target.cpu().numpy(), :] = before_lincls_feat_t.cpu().detach().numpy()
                            pseudo_label_t_cls = y_target
                            pseudo_label_t_ot = y_target

                            pseudo_label_t_cls = torch.argmax(after_lincls_t, axis=1)

                            norm_feat_t = F.normalize(before_lincls_feat_t)
                            sim_tar_all = torch.matmul(norm_feat_t, source_prototype.t())
                            _, _, _, wt_i = ubot_CCD(
                                sim_tar_all, beta, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter
                            )
                            confidence_ot_t = (wt_i - torch.min(wt_i)) / (torch.max(wt_i) - torch.min(wt_i) + 1e-12)

                            Escore_t, entropy_cls_t, softmax_cls_t, confidence_ot_t = ensemble_score(
                                after_lincls_t, confidence_ot_t
                            )

                            if threshold_test == None:
                                rand_n = min(mixup_samples, len(z_target))
                                idx_rand_test = torch.randperm(len(z_target), device=self.device)[:rand_n]
                                ubot_feature_t_mix = mixup_target(z_target[idx_rand_test], device=self.device)
                                before_lincls_feat_t_mix, logit_t_mix = self.classifier(ubot_feature_t_mix)
                                norm_feat_t_mix = F.normalize(before_lincls_feat_t_mix)
                                sim_tar_mix = torch.matmul(norm_feat_t_mix, source_prototype.t())
                                _, _, _, wt_i_mix = ubot_CCD(
                                    sim_tar_mix, beta_s, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter
                                )
                                confidence_ot_mix = (wt_i_mix - torch.min(wt_i_mix)) / (
                                    torch.max(wt_i_mix) - torch.min(wt_i_mix) + 1e-12
                                )
                                Escore_mix_t, _, _, _ = ensemble_score(logit_t_mix, confidence_ot_mix)
                                threshold_test = torch.mean(Escore_mix_t)
                                threshold_test = torch.min(
                                    torch.tensor([threshold_test, torch.quantile(Escore_mix_t, (1 - resolution))], device=self.device)
                                )

                            pred_known_id = torch.nonzero(Escore_t >= threshold_test).view(-1)
                            pred_unknown_id = torch.nonzero(Escore_t < threshold_test).view(-1)
                            pseudo_label_t_cls[pred_unknown_id] = -1
                            pseudo_label_t_ot[pred_unknown_id] = -1
                            ##先分known和unknown 对known做ot 避免unknown的样本加进去不符合beta的分布
                            sim_tar_high_conf = torch.matmul(norm_feat_t[pred_known_id], source_prototype.t())
                            if sim_tar_high_conf.size()[0] > 0:
                                pseudo_label_t_high_conf, _, _, _ = ubot_CCD(
                                    sim_tar_high_conf, beta, device=self.device, stopThr=sinkhorn_stop_thr, max_iter=sinkhorn_max_iter
                                )
                                pseudo_label_t_ot[pred_known_id] = pseudo_label_t_high_conf

                            idx_target = id_target.cpu().numpy()
                            pred_label_bank_cls[idx_target] = pseudo_label_t_cls.cpu().detach().numpy()
                            pred_label_bank_ot[idx_target] = pseudo_label_t_ot.cpu().detach().numpy()
                            Escore_bank[idx_target] = Escore_t.cpu().detach().numpy()
                            entropy_cls_bank[idx_target] = entropy_cls_t.cpu().detach().numpy()
                            softmax_cls_bank[idx_target] = softmax_cls_t.cpu().detach().numpy()
                            confidence_ot_bank[idx_target] = confidence_ot_t.cpu().detach().numpy()
                            target_known_cnts += len(index_known)
                            target_unknown_cnts += len(index_unknown)

                        if iteration == miditer1 and novel_cell_test:
                            dip_test = dip(Escore_bank, num_bins=100)
                            dip_p_value = dip_test[-1]
                            dip_test = dip_test[-2]
                            ensemble_total_scale = np.sqrt(Escore_bank)
                            BC = calculate_bimodality_coefficient(ensemble_total_scale)
                            print("bimodality of dip test:", dip_p_value, not dip_test)
                            print("bimodality coefficient:(>0.45 indicates bimodality)", BC, BC > 0.45)
                            bimodality = (not dip_test) or (BC > 0.45)
                            print("ood sample exists:", bimodality)
                            if bimodality:
                                resolution = 0.5
                            else:
                                resolution = 0.8

                        index_tf_known_all = [
                            target_labels[i] in list(np.unique(source_labels)) for i in range(len(target_labels))
                        ]
                        index_known_all = [i for i in range(len(index_tf_known_all)) if index_tf_known_all[i]]
                        index_unknown_all = [i for i in range(len(index_tf_known_all)) if not index_tf_known_all[i]]
                        target_labels[index_unknown_all] = -1
                        Escore_known = Escore_bank[index_known_all]
                        Escore_unknown = Escore_bank[index_unknown_all]

                        entropy_cls_known = entropy_cls_bank[index_known_all]
                        entropy_cls_unknown = entropy_cls_bank[index_unknown_all]

                        softmax_cls_known = softmax_cls_bank[index_known_all]
                        softmax_cls_unknown = softmax_cls_bank[index_unknown_all]

                        confidence_ot_known = confidence_ot_bank[index_known_all]
                        confidence_ot_unknown = confidence_ot_bank[index_unknown_all]

                        target_known_cnts = len(index_known_all)
                        target_known_corrcnts_ot = np.sum(
                            pred_label_bank_ot[index_known_all] == target_labels[index_known_all]
                        )
                        target_known_corrcnts_cls = np.sum(
                            pred_label_bank_cls[index_known_all] == target_labels[index_known_all]
                        )
                        target_unknown_cnts = len(index_unknown_all)
                        target_unknown_corrcnts_ot = np.sum(
                            pred_label_bank_ot[index_unknown_all] == target_labels[index_unknown_all]
                        )
                        target_unknown_corrcnts_cls = np.sum(
                            pred_label_bank_cls[index_unknown_all] == target_labels[index_unknown_all]
                        )
                        acc_source_ot = source_corrcnts_ot / source_cnts
                        acc_source_cls = source_corrcnts_cls / source_cnts
                        acc_ot = (target_known_corrcnts_ot + target_unknown_corrcnts_ot) / (
                            target_known_cnts + target_unknown_cnts
                        )
                        acc_cls = (target_known_corrcnts_cls + target_unknown_corrcnts_cls) / (
                            target_known_cnts + target_unknown_cnts
                        )
                        acc_known_ot = target_known_corrcnts_ot / target_known_cnts
                        acc_unknown_ot = target_unknown_corrcnts_ot / target_unknown_cnts
                        acc_known_cls = target_known_corrcnts_cls / target_known_cnts
                        acc_unknown_cls = target_unknown_corrcnts_cls / target_unknown_cnts
                        print(
                            "#Iter %d: (OT) Source acc: %f, Target total acc : %f,  Target known acc: %f, Target unknown acc: %f"
                            % (iteration, acc_source_ot, acc_ot, acc_known_ot, acc_unknown_ot)
                        )
                        print(
                            "#Iter %d: (CLS) Source acc: %f, Target total acc : %f,  Target known acc: %f, Target unknown acc: %f"
                            % (iteration, acc_source_cls, acc_cls, acc_known_cls, acc_unknown_cls)
                        )

        adata_source = sc.AnnData(sp.csr_matrix(source_z_bank))
        adata_source.obs["cell_type"] = [cell_types[i] for i in source_labels.astype(int)]
        adata_source.obs["source"] = "scRNA"

        adata_target = sc.AnnData(sp.csr_matrix(target_z_bank))
        total_cell_types = list(common_cell_type) + ["Unknown"]
        adata_target.obs["cell_type"] = [cell_types[i] for i in target_labels.astype(int)]
        adata_target.obs["pred_cls"] = [total_cell_types[i] for i in pred_label_bank_cls.astype(int)]
        adata_target.obs["pred_ot"] = [total_cell_types[i] for i in pred_label_bank_ot.astype(int)]
        adata_target.obs[["X", "Y"]] = np.array(spatial_coor)
        adata_target.obs["Escore"] = Escore_bank
        adata_target.obs["source"] = "Spatial"

        return adata_source, adata_target, threshold_test

    # Aliases
    fit = train
    fit_eval = train_eval
