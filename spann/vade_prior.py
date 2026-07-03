import torch
import torch.nn as nn
import math

class GMMPrior(nn.Module):
    def __init__(self, latent_dim, num_known_classes):
        super(GMMPrior, self).__init__()
        self.num_known_classes = num_known_classes
        self.latent_dim = latent_dim

        # CHỈ khai báo tham số cho K cụm đã biết
        self.pi = nn.Parameter(torch.ones(num_known_classes + 1) / (num_known_classes + 1))
        self.mu_c = nn.Parameter(torch.randn(num_known_classes, latent_dim))
        self.logvar_c = nn.Parameter(torch.zeros(num_known_classes, latent_dim))

    def forward(self, z, mu, logvar, labels=None):
        batch_size = z.size(0)
        log_pi = torch.log_softmax(self.pi, dim=0)

        # TẠO CỤM BACKGROUND: Cố định mu=0, logvar=0 (Tương đương var=1)
        bg_mu = torch.zeros(1, self.latent_dim, device=z.device)
        bg_logvar = torch.zeros(1, self.latent_dim, device=z.device)

        # Ghép Cụm Background vào cuối danh sách cụm
        full_mu_c = torch.cat([self.mu_c, bg_mu], dim=0)
        full_logvar_c = torch.cat([self.logvar_c, bg_logvar], dim=0)
        full_var_c = torch.exp(full_logvar_c)

        # Mở rộng chiều để tính toán Broadcasting
        z_expand = z.unsqueeze(1)
        mu_c_expand = full_mu_c.unsqueeze(0)
        var_c_expand = full_var_c.unsqueeze(0)

        log_p_z_c = -0.5 * torch.sum(
            torch.log(2 * math.pi * var_c_expand) + 
            ((z_expand - mu_c_expand) ** 2) / var_c_expand, 
            dim=2
        )
        log_p_c_z = log_pi + log_p_z_c
        
        if labels is not None:
            # Nếu có nhãn (Source), ép vào K cụm đầu tiên
            gamma = torch.zeros(batch_size, self.num_known_classes + 1, device=z.device)
            gamma.scatter_(1, labels.unsqueeze(1), 1.0)
        else:
            # Nếu không có nhãn (Target), cho phép dạt vào cụm Background
            gamma = torch.softmax(log_p_c_z, dim=1) 

        kl_discrete = torch.sum(gamma * (torch.log(gamma + 1e-10) - log_pi), dim=1)

        mu_expand = mu.unsqueeze(1)
        logvar_expand = logvar.unsqueeze(1)
        var_expand = torch.exp(logvar_expand)

        # Tính thành phần liên tục với danh sách cụm mở rộng (full_*)
        kl_continuous = 0.5 * torch.sum(
            gamma * torch.sum(
                full_logvar_c.unsqueeze(0) - logvar_expand + 
                (var_expand + (mu_expand - mu_c_expand) ** 2) / var_c_expand - 1, 
                dim=2
            ), 
            dim=1
        )

        loss_kl_gmm = torch.mean(kl_discrete + kl_continuous)
        return loss_kl_gmm, gamma
