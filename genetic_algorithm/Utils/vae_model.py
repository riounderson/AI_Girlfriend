import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Internal State を VAE の潜在変数として管理するための VAE モデル
    """
    def __init__(self, input_dim=4, latent_dim=4):  # 4 次元入力に修正
        super(VAE, self).__init__()
        
        # エンコーダ（入力を潜在変数へ圧縮）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),  
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # 出力サイズ = 2 * latent_dim (例: 8)
        )
        
        # デコーダ（潜在変数を元の形式に復元）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        print(f"エンコーダ出力の shape: {h.shape}")  # デバッグ用
        
        if h.dim() == 1:  # 1D テンソルの場合、2D に変換
            h = h.unsqueeze(0)  # (8,) → (1, 8)

        mu, logvar = torch.chunk(h, 2, dim=1)  # 2つの部分に分割
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
