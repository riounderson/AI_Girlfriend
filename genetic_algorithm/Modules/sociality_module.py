from Utils.internal_state import InternalState
from Utils.invoke_llm import get_claude_response
from Modules.working_memory import WorkingMemory
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class SocialityModule:
    """
    社会的満足度（sociality）を評価して内部状態に登録するモジュール。
    - 会話頻度が少ない
    - 応答がない
    - 会話内容に孤独感・拒絶感が含まれる
    などを考慮して評価する。
    """

    def __init__(self):
        self.internal_state = InternalState()
        self.working_memory = WorkingMemory()

    def calculate_sociality(self):
        all_memories = self.working_memory.get_memory()
        if not all_memories:
            print("⚠️ Working Memory が空です。sociality を低く評価します。")
            self.internal_state.modify_state_value("sociality", -0.9)
            return -0.9

        now = datetime.now(ZoneInfo("Asia/Tokyo"))
        threshold_minutes = 1

        # ① 指定期間内のメモリだけに絞り込む
        cutoff = now - timedelta(minutes=threshold_minutes)
        recent_memories = [
            e for e in all_memories
            if datetime.fromisoformat(e["timestamp"]) >= cutoff
        ]

        # ② 相手（others）の発話だけカウント
        recent_count = sum(1 for e in recent_memories if e.get("speaker") == "others")

        # 応答性: ユーザー発話とそれに続く自分の応答数
        user_turns = [e for e in recent_memories if e["speaker"] == "others"]
        self_responses = [e for e in recent_memories if e["speaker"] == "self"]
        # response_ratio = len(self_responses) / max(1, len(user_turns))

        # 会話のテキスト部分（話者付き）
        dialogue_str = "\n".join(
            f"{'自分' if e['speaker'] == 'self' else '相手'}: {e['text']}"
            for e in recent_memories
        )

        # prompt = f"""
        # 以下の会話履歴を元に、AI（彼女）の「社会的満足度（sociality）」を -1.0 〜 1.0 のスケールで評価してください。
        # 以下のような会話を自分自身が行ったとしてどのくらい社会的満足度が充足されたかという観点で評価してください。
        # ただし、あなたは寂しがりやなので少し厳しめに評価をお願いします。また会話頻度の方を重視して採点してください。
        # **なお評価の理由など余計な情報は一切いりません。点数だけを出力してください。

        # 【会話頻度（{threshold_minutes}分以内）】: {recent_count} 回
        # 【応答率（彼氏の発話に対して彼女が返した割合）】: {response_ratio:.2f}

        # 【会話履歴】:
        # {dialogue_str}

        # 【評価指針】
        # - 無視された・孤独・会話が途切れた・長い間誰とも喋ってないなど人との繋がりを感じていない場合 → -1.0 に近い
        # - 会話が続いた・肯定的・共感があったなど人との繋がりを感じた場合 → 1.0 に近い

        # 【出力】
        # 数値のみ（例: -0.8, 0.5 など）
        # """

        prompt = f"""
        以下の会話履歴を元に、AI（彼女）の「社会的満足度（sociality）」を -1.0 〜 1.0 のスケールで評価してください。
        以下のような会話を自分自身が行ったとしてどのくらい社会的満足度が充足されたかという観点で評価してください。
        ただし、あなたは寂しがりやなので少し厳しめに評価をお願いします。また会話頻度の方を重視して採点してください。
        **なお評価の理由など余計な情報は一切いりません。点数だけを出力してください。

        【会話頻度（{threshold_minutes}分以内）】: {recent_count} 回

        【会話履歴】:
        {dialogue_str}

        【評価指針】
        - 無視された・孤独・会話が途切れた・長い間誰とも喋ってないなど人との繋がりを感じていない場合 → -1.0 に近い
        - 会話が続いた・肯定的・共感があったなど人との繋がりを感じた場合 → 1.0 に近い

        【出力】
        数値のみ（例: -0.8, 0.5 など）
        """

        response = get_claude_response(prompt)

        try:
            score = float(response.strip())
            score = max(-1.0, min(1.0, score))
            self.internal_state.modify_state_value("sociality", score)
            return score
        except ValueError:
            print(f"⚠️ sociality の評価に失敗: {response}")
            self.internal_state.modify_state_value("sociality", 0.0)
            return 0.0
