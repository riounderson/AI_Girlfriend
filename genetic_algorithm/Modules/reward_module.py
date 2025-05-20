# Modules/reward_module.py

from short_term_memory import STM
from Utils.internal_state import InternalState
from Modules.working_memory import WorkingMemory
from Utils.invoke_llm import get_claude_response

class RewardModule:
    def __init__(self):
        self.internal_state = InternalState()
        self.wm = WorkingMemory()
        self.stm = STM()

    def evaluate_reward(self, purpose: str = None):
        recent_memory = self.wm.get_memory()
        if not recent_memory or not purpose:
            print("⚠️ 評価対象がありません（目的または記憶が空）")
            return

        recent_texts = "\n".join([entry["text"] for entry in recent_memory[-3:]])

        prompt = f"""
        あなたは目的を持って相手と会話しています。以下の目的と直近の会話を基に、目的達成度（-1.0〜1.0）を評価してください。
        **なお評価の理由など余計な情報は一切いりません。点数だけを出力してください。

        【目的】
        {purpose}

        【会話】
        {recent_texts}
        """

        try:
            response = get_claude_response(prompt)
            print(f"responseでええエス: {response}")
            reward_score = float(response.strip())
            reward_score = max(-1.0, min(1.0, reward_score))
        except Exception as e:
            print(f"⚠️ LLM 評価失敗: {e}")
            reward_score = 0.0

        print(f"🏆 報酬スコア（目的達成度）: {reward_score:.2f}")
        self.internal_state.modify_state_value("short_reward", reward_score)

        # ⬇️ 高報酬だった場合は STM に記憶する
        if reward_score >= 0.7:
            self.stm.add_learning_result(
                purpose=purpose,
                feedback=recent_texts,
                reward=reward_score
            )
    
    def evaluate_longtime_reward(self):
        goal = self.internal_state.retrieve_text_state("long_term_goal")
        recent_memory = self.wm.get_memory()
        if not recent_memory or not goal:
            print("⚠️ 評価対象がありません（目的または記憶が空）")
            return

        recent_texts = "\n".join([entry["text"] for entry in recent_memory[-8:]])

        prompt = f"""
        あなたは対人関係において以下のような長期的な目標を策定して行動をしています。直近の会話の内容から目標達成度（-1.0〜1.0）を評価してください。
        会話内容から見て完全に目標が達成されていると思われる場合: 1.0
        ex) 好意を得たい人から「好きだ」や好意を伝えらるときなど
        特に判断できない場合は0
        ex) 「こんにちは」などの単なる日常会話で特に会話内容が長期的目標と関係しないと思われる場合
        完全に失敗している場合は-1.0
        ex) 好意を得たい人から敵意を向けられる発言をされるなど
        としてください。

        【絶対守るべきルール】
        ・評価の理由など余計な情報は一切いりません。点数だけを出力すること

        【長期的な目標】
        {goal}

        【会話】
        {recent_texts}
        """

        try:
            response = get_claude_response(prompt)
            long_reward_score = float(response.strip())
            long_reward_score = max(-1.0, min(1.0, long_reward_score))
        except Exception as e:
            print(f"⚠️ LLM 評価失敗: {e}")
            long_reward_score = 0.0

        print(f"🏆 報酬スコア（目標達成度）: {long_reward_score:.2f}")
        self.internal_state.modify_state_value("long_term_reward", long_reward_score)

        # ⬇️ 高報酬だった場合は STM に記憶する
        if long_reward_score >= 0.7:
            self.stm.add_learning_result(
                purpose=goal,
                feedback=recent_texts,
                reward=long_reward_score
            )

    
