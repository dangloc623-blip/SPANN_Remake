import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class VaDEPrior(nn.Module):
    def __init__(self, n_components, latent_dim):
        super(VaDEPrior, self).__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim

        # Khởi tạo tham số GMM: Xác suất chọn cụm (pi), Trung bình (mu), Phương sai (log_var)
        self.pi_logits = nn.Parameter(torch.ones(n_components))
        self.mu_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.log_var_c = nn.Parameter(torch.zeros(n_components, latent_dim))

    def get_pi(self):
        # Tránh pi = 0 gây lỗi NaN khi tính log
        return F.softmax(self.pi_logits, dim=0)

    def compute_gamma(self, z):
        """ Tính xác suất hậu nghiệm p(c|z) - Soft assignment """
        z = z.unsqueeze(1) # [Batch, 1, Dim]
        mu = self.mu_c.unsqueeze(0) # [1, Clusters, Dim]
        log_var = self.log_var_c.unsqueeze(0)
        var_c = torch.exp(log_var) + 1e-8

        # log p(z|c)
        log_p_z_c = -0.5 * torch.sum(
            log_var + (z - mu)**2 / var_c + math.log(2 * math.pi), dim=2
        )
        
        # log p(c|z) tỷ lệ thuận với log( p(z|c) * p(c) )
        log_pi = F.log_softmax(self.pi_logits, dim=0).unsqueeze(0)
        log_p_c_z = log_p_z_c + log_pi

        # Chuẩn hóa bằng Softmax để ra gamma (xác suất thuộc từng cụm)
        gamma = F.softmax(log_p_c_z, dim=1)
        return gamma

    def compute_vade_loss(self, z, mu_z, var_z):
        """ Tính toán KL Divergence cho GMM Prior """
        gamma = self.compute_gamma(z)
        
        mu_z = mu_z.unsqueeze(1) 
        var_z = var_z.unsqueeze(1)
        mu_c = self.mu_c.unsqueeze(0) 
        var_c = torch.exp(self.log_var_c).unsqueeze(0) + 1e-8

        # 1. KL( q(z|x) || p(z|c) )
        kl_z = 0.5 * torch.sum(
            torch.log(var_c) - torch.log(var_z + 1e-10) + 
            (var_z + (mu_z - mu_c)**2) / var_c - 1, dim=2
        )

        # 2. KL( q(c|x) || p(c) )
        log_pi = F.log_softmax(self.pi_logits, dim=0).unsqueeze(0)
        kl_c = torch.sum(gamma * (torch.log(gamma + 1e-10) - log_pi), dim=1)

        # Tổng hợp VaDE Loss
        loss = torch.sum(gamma * kl_z, dim=1) + kl_c
        return loss.mean()
