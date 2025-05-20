import numpy as np
from Modules.working_memory import WorkingMemory
from Utils.internal_state import InternalState
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class CommonSenseModule:
    """
    会話において「ユーザーに話しかけられたのに応答しなかった場合」にストレスとみなすモジュール。
    応答の放置 = 社会的常識に反する = common_sense を低下させる。
    """

    def __init__(self):
        self.working_memory = WorkingMemory()
        self.internal_state = InternalState()

    def evaluate_common_sense(self):
        """
        直近で user → self の発話があり、その後 self の応答がなかった場合に common_sense を下げる。
        """
        memory = self.working_memory.get_memory()
        if not memory:
            self.internal_state.modify_state_value("common_sense", 0.0)
            return 0.0

        # 最後の「ユーザーからの発話」を見つける
        last_user_said = None
        for entry in reversed(memory):
            if entry.get("speaker") == "others":
                last_user_said = entry
                break

        if not last_user_said:
            self.internal_state.modify_state_value("common_sense", 0.0)
            return 0.0

        # 最後の user 発話の timestamp を取得
        last_user_time = datetime.fromisoformat(last_user_said["timestamp"])

        # それ以降に self（彼女）の発話があったか調べる
        replied = False
        for entry in reversed(memory):
            if entry["timestamp"] <= last_user_said["timestamp"]:
                break  
            if entry.get("speaker") == "self":
                replied = True
                break

        # 応答がなければ common_sense を低下させる
        if not replied:
            print("⚠️ ユーザーの発話を無視してしまいました。common_sense を -0.8 に設定します。")
            self.internal_state.modify_state_value("common_sense", -0.9)
            return -0.8
        else:
            self.internal_state.modify_state_value("common_sense", 0.0)
            return 0.0
