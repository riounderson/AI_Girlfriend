
"""
短期目的 (purpose) ↔ 会話ペア を評価し
  • InternalState.short_reward を更新
  • 高スコア事例を STM に保存
Girlfriend.run_module_loop() から定期的に呼び出すことを想定
"""

import time
from typing import List, Dict, Tuple, Optional

from Utils.internal_state import InternalState
from Modules.working_memory import WorkingMemory
from short_term_memory import STM
from Utils.invoke_llm import get_claude_response


class ShortRewardModule:
    # ── ハイパーパラメータ ───────────────────────────── #
    MAX_WAIT_SEC       = 30     # 自発発話後、相手応答を待つ最大秒数
    REWARD_THRESHOLD   = 0.70   # STM へ格納する報酬閾値
    MEMORY_WINDOW      = 3      # 最近 N 件のペアを保持
    # -------------------------------------------------- #

    def __init__(self):
        self.internal_state = InternalState()
        self.wm             = WorkingMemory()
        self.stm            = STM()

        # 「どこまでスキャン済みか」を覚えておくポインタ
        self._last_idx: int = 0

    # -------------------------------------------------- #
    # Public API  ―  Girlfriend から呼ばれる関数はこれだけ
    # -------------------------------------------------- #
    def evaluate_shortterm_reward(self) -> None:
        """
        • WorkingMemory を _last_idx 以降だけ走査  
        • (self 発話 with purpose) → (user 応答) ペアが揃ったら採点  
        • short_reward 更新 & 高スコアなら STM 保存  
        """
        mem: List[Dict] = self.wm.get_memory()
        i = self._last_idx

        while i < len(mem) - 1:          # 少なくとも自分＋相手で2行必要
            self_entry = mem[i]

            # 🎯 自分発話 かつ purpose 付き？
            if self_entry.get("speaker") == "self" and self_entry.get("purpose"):
                purpose = self_entry["purpose"]

                # 次の user 発話を探す
                j, user_entry = self._find_next_user(mem, i + 1)
                if user_entry:
                    self._score_and_store(purpose, self_entry, user_entry)
                    i = j + 1           # 次へ
                else:
                    # 相手応答待ち。次回呼び出し時に再判定
                    break
            else:
                i += 1

        self._last_idx = i    # 走査済み位置を更新

    # -------------------------------------------------- #
    # internal helpers
    # -------------------------------------------------- #
    def _find_next_user(self, mem: List[Dict], start: int) -> Tuple[int, Optional[Dict]]:
        """
        start 以降で最初の user 発話を返す。  
        • 応答が無い / 古すぎる場合は (idx, None)
        """
        now = time.time()
        for idx in range(start, len(mem)):
            entry = mem[idx]
            if entry.get("speaker") == "user":
                ts = entry.get("timestamp", now)
                if (now - ts) <= self.MAX_WAIT_SEC:
                    return idx, entry         # ⏱ 期限内
                return idx, None               # ⌛ 遅すぎる
        return len(mem), None                  # 見つからず

    def _score_and_store(self, purpose: str, self_e: Dict, user_e: Dict) -> None:
        """LLM で採点 → short_reward 反映 → STM 保存(必要なら)"""
        dialogue = f"あなた: {self_e['text']}\n相手: {user_e['text']}"

        prompt = f"""
        以下の【目的】と【対話】を踏まえ、
        目的達成度を -1.0〜1.0 の数値だけで返答してください。

        【目的】
        {purpose}

        【対話】
        {dialogue}
        """

        try:
            score = float(get_claude_response(prompt).strip())
            score = max(-1.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️ RewardModule: LLM 評価失敗 ({e})")
            score = 0.0

        print(f"🏆 short_term_reward = {score:.2f}  (purpose: {purpose})")
        self.internal_state.modify_state_value("short_term_reward", score)

        if score >= self.REWARD_THRESHOLD:
            self.stm.add_learning_result(
                purpose=purpose,
                feedback=dialogue,
                reward=score
            )
