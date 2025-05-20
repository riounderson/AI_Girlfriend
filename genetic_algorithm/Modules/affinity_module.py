import numpy as np
from Utils.manage_memory import ChromaMemory
from Utils.invoke_llm import get_claude_response  # Bedrock API を呼び出す関数
from Utils.internal_state import InternalState

class AffinityModule:
    """
    彼女AIの親密度（Affinity）を評価し、Internal State に登録するモジュール
    """
    def __init__(self):
        self.stm = ChromaMemory(db_path="memory-db/STM", collection_name="STM_memory")
        self.internal_state = InternalState()  # InternalState の参照

    def evaluate_affinity_with_llm(self, user_input):
        """
        LLM を利用してユーザーの発話から親密度を評価
        """
        prompt = f"""
        ユーザーの発話から親密度（-1 〜 1 のスコア）を評価してください。
        なお回答は回答例のように数字だけで回答してください。
        -1: 非常に冷たい・敵対的な発言
         0: 中立的な発言
         1: 非常に親密な発言

        【回答例】
        0.5
        0.4

        ユーザーの発話: "{user_input}"
        """
        response = get_claude_response(prompt)

        # LLM の出力を数値化
        try:
            affinity_score = float(response.strip())
            affinity_score = max(-1, min(1, affinity_score))  # -1 〜 1 の範囲に制限
        except ValueError:
            affinity_score = 0  # エラー時は中立とする

        return affinity_score

    def calculate_affinity(self, user_input=None):
        """
        LLM を使って親密度を評価し、Internal State に登録
        """
        # 最近の会話頻度を取得
        recent_conversations = len(self.stm.collection.get()["documents"])

        # 会話頻度をスコア化（10回以上の会話なら最大値に近づく）
        frequency_score = np.tanh(recent_conversations / 10)  # -1 〜 1 に収める

        # LLM を使って親密度を評価
        llm_affinity = self.evaluate_affinity_with_llm(user_input)

        # 最終的な親密度スコア（会話頻度と LLM の評価を組み合わせる）
        final_affinity = (0.7 * llm_affinity) + (0.3 * frequency_score)

        # Internal State に登録
        self.internal_state.modify_state_value("affinity", final_affinity)

        return round(final_affinity, 2)
