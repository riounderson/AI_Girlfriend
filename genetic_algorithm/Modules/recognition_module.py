import os
from Modules.working_memory import WorkingMemory
from Utils.invoke_llm import get_claude_response
from Utils.internal_state import InternalState

class RecognitionModule:
    """
    自己イメージと現実のギャップを評価するモジュール
    """

    def __init__(self):
        self.internal_state = InternalState()
        self.working_memory = WorkingMemory()

        # 自己イメージを設定ファイル（txt）からロード
        self.self_image = self.load_self_image()

    def load_self_image(self, filepath="/Users/riyon/WorkSpaces/Development/GenAI_Girlfriend/workspace/Gen_AI_Girl_Friend/src/genetic_algorithm/self_image.txt"):
        """自己イメージ（理想の見られ方）をロード"""
        if not os.path.exists(filepath):
            print(f"⚠️ 自己イメージ設定ファイルが見つかりません: {filepath}")
            return "私は思いやりがあり、信頼され、愛される存在でありたい。"

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content

    def calculate_recognition(self):
        """
        自己イメージと最近の会話内容から、認識ギャップを評価し、Internal State に登録する
        """
        recent_memories = self.working_memory.get_memory()

        if not recent_memories:
            print("⚠️ WorkingMemory に会話履歴がないため、認識評価スキップ")
            self.internal_state.modify_state_value("recognition", 0.0)
            return 0.0

        # 直近3件程度を対象にする
        recent_texts = "\n".join([entry["text"] for entry in recent_memories[-8:]])

        prompt = f"""
        あなたは AI です。
        以下の「自己イメージ」と「直近の会話履歴」を比較して、自己イメージにどれだけ近い扱いを受けているかを -1.0〜1.0 のスコアで評価してください。
        **なお評価の理由など余計な情報は一切いりません。点数だけを出力してください。

        【自己イメージ】
        {self.self_image}

        【直近の会話履歴】
        {recent_texts}

        【スコア基準】
        -1.0 = 自己イメージと大きくかけ離れている
         0.0 = どちらともいえない（特に判断が迷うような場合はこちらの点数で大丈夫です。なぜなら日常会話のほとんどは自分が相手からどのように見られているのか判断するのに適さないからです。）
        +1.0 = 自己イメージに非常に近い扱いを受けている

        【回答例】
        0.7
        """

        response = get_claude_response(prompt)

        try:
            recognition_score = float(response.strip())
            recognition_score = max(-1.0, min(1.0, recognition_score))
        except ValueError:
            recognition_score = 0.0  # エラー時は無難に中立

        self.internal_state.modify_state_value("recognition", recognition_score)

        print(f"🧠 [Recognition] 自己イメージ適合度: {recognition_score:.2f}")
        return recognition_score
