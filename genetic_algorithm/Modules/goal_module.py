from Utils.invoke_llm import get_claude_response
from Utils.internal_state import InternalState

class GoalModule:
    """
    内部状態を参照して中長期的な目標（テキスト）を生成・登録するモジュール
    """

    def __init__(self):
        self.internal_state = InternalState()

    def generate_goal(self):
        # 数値情報の内部状態を取得
        current_states = self.internal_state.get_named_state()

        prompt = f"""
        あなたはAIとして自律的に行動しようとしています。
        以下はあなたの現在の内部状態（各スコアは -1〜1 の範囲で表現されています）です：

        {current_states}

        この内部状態を改善し、より良好な状態へ向かうために、中長期的に達成すべき目標を1つ提案してください。
        例えば、内部状態の孤独感が高い→誰かと親密な関係を築こう、のような感じです。
        出力はその目標のテキストのみで、説明は不要です。
        """

        # print(f"goalのプロンプトだよん: {prompt}")
        response = get_claude_response(prompt)
        return response.strip()

    def update_internal_goal(self):
        """
        生成した中長期的目標を内部状態に文字列として登録する
        """
        goal_text = self.generate_goal()
        # goal_text = "test"
        self.internal_state.modify_state_text("long_term_goal", goal_text)
        print(f"🎯 新たな目標を内部状態に登録しました: {goal_text}")
