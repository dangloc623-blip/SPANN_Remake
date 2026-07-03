import torch
import torch.nn as nn
import math

class GMMPrior(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        """
        Module tính toán Loss dựa trên GMM Prior thay thế cho standard Gaussian.
        
        Args:
            latent_dim (int): Số chiều của không gian tiềm ẩn z.
            n_clusters (int): Số lượng cụm tế bào dự kiến (K).
        """
        super(GMMPrior, self).__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        # Khởi tạo các tham số học được của GMM (Bước M sẽ được tối ưu qua Gradient Descent)
        # pi: Xác suất tiên nghiệm của mỗi cụm (phân phối Categorical)
        self.pi = nn.Parameter(torch.ones(n_clusters) / n_clusters)
        # mu_c: Kỳ vọng của mỗi cụm
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))
        # var_c: Phương sai của mỗi cụm (lưu ý: để tối ưu tốt hơn, thường giữ log_var thay vì var)
        self.logvar_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def forward(self, z, mu, logvar, labels=None):
        batch_size = z.size(0)
        log_pi = torch.log_softmax(self.pi, dim=0)
        var_c = torch.exp(self.logvar_c)

        z_expand = z.unsqueeze(1)
        mu_c_expand = self.mu_c.unsqueeze(0)
        var_c_expand = var_c.unsqueeze(0)

        log_p_z_c = -0.5 * torch.sum(
            torch.log(2 * math.pi * var_c_expand) + 
            ((z_expand - mu_c_expand) ** 2) / var_c_expand, 
            dim=2
        )
        log_p_c_z = log_pi + log_p_z_c
        
        if labels is not None:
            # [BÁN GIÁM SÁT] Neo cứng xác suất bằng nhãn thật cho tập Source
            gamma = torch.zeros(batch_size, self.n_clusters, device=z.device)
            # labels.unsqueeze(1) chứa index từ 0 đến K-1
            gamma.scatter_(1, labels.unsqueeze(1), 1.0)
        else:
            # [PHI GIÁM SÁT] Tự động phân cụm cho tập Target (có cơ hội rơi vào cụm K+1)
            gamma = torch.softmax(log_p_c_z, dim=1) 

        kl_discrete = torch.sum(gamma * (torch.log(gamma + 1e-10) - log_pi), dim=1)

        mu_expand = mu.unsqueeze(1)
        logvar_expand = logvar.unsqueeze(1)
        var_expand = torch.exp(logvar_expand)

        kl_continuous = 0.5 * torch.sum(
            gamma * torch.sum(
                self.logvar_c.unsqueeze(0) - logvar_expand + 
                (var_expand + (mu_expand - mu_c_expand) ** 2) / var_c_expand - 1, 
                dim=2
            ), 
            dim=1
        )

        loss_kl_gmm = torch.mean(kl_discrete + kl_continuous)
        return loss_kl_gmm, gamma
