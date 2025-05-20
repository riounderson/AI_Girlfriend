import torch
import numpy as np
from Utils.vae_model import VAE
import threading
from sentence_transformers import SentenceTransformer

class InternalState:
    """
    Internal State を管理するクラス。
    数値情報は VAE に渡して潜在ベクトルとして評価、
    テキスト情報はそのまま保存・参照される。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InternalState, cls).__new__(cls)
        return cls._instance
    def __init__(self):
        self.vae = VAE(input_dim=5, latent_dim=5)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.state_vector = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float32).unsqueeze(0)

        # 数値評価用の状態（VAEに入力する）
        self.state_values = {
            "common_sense": 0.5,
            "recognition": 0.5,
            "short_reward": 0.5,
            "long_term_reward": 0.5,
            "sociality": 0.5
        }

        # 意味的なテキスト状態（目標や方針など）
        self.state_texts = {
            "long_term_goal": {
                "text": "",
                "embedding": None
            }
        }

    # ===== 数値状態の操作 =====

    def modify_state_value(self, name, value):
        """数値情報を更新"""
        if name in self.state_values:
            self.state_values[name] = np.clip(value, -1, 1)
            print(f"🔄 State 更新: {name} = {self.state_values[name]}")
        else:
            print(f"⚠️ `{name}` は存在しません。")

    def get_named_state(self):
        """全数値状態を辞書形式で返す"""
        return self.state_values.copy()

    def update_all_states(self):
        """VAE に数値状態を入力し、潜在ベクトルに変換"""
        input_tensor = torch.tensor(
            list(self.state_values.values()), dtype=torch.float32
        ).unsqueeze(0)

        _, mu, logvar = self.vae(input_tensor)
        self.state_vector = self.vae.reparameterize(mu, logvar)

    def get_internal_state(self):
        """潜在ベクトルを返す（評価用）"""
        return self.state_vector.detach().numpy().flatten()

    # ===== テキスト状態の操作 =====

    def modify_state_text(self, name, text):
        """意味的なテキスト情報を更新（embedding含む）"""
        embedding = self.sentence_transformer.encode(text)
        self.state_texts[name] = {
            "text": text,
            "embedding": embedding
        }
        print(f"📝 Text State 更新: {name} = '{text}'")

    def retrieve_text_state(self, name):
        """テキスト状態を取得"""
        if name not in self.state_texts or not self.state_texts[name]["text"]:
            print(f"⚠️ `{name}` にテキストが登録されていません")
            return None
        return self.state_texts[name]["text"]

    def get_text_embedding(self, name):
        """テキスト状態のベクトルを取得（オプション）"""
        if name not in self.state_texts or self.state_texts[name]["embedding"] is None:
            print(f"⚠️ `{name}` にベクトルが登録されていません")
            return None
        return self.state_texts[name]["embedding"]
