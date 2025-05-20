import time
import random
import threading
import numpy as np
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.chat_models import BedrockChat
from Utils.internal_state import InternalState
from actions import ActionManager

class CognitiveController:
    """
    彼女AIの自我を司るクラス。
    Internal State を VAE の潜在変数として管理し、意思決定に活用する。
    """
    _instance = None  # シングルトン用のクラス変数
    _lock = threading.Lock()  # スレッドセーフなロック
    _client_socket = None  # `client_socket` もシングルトンにする

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CognitiveController, cls).__new__(cls)
                cls._instance._initialized = False  # 初期化チェック用
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.internal_state = InternalState()
            self.ideal_state = np.array([0.8, 0.8, 0.8, 0.8])
            self.feature_names = ["mood", "affinity", "interest", "conversation"]
            self.action_manager = ActionManager() 
            self.agent = self.initialize_agent()
            self._initialized = True  # 初期化フラグ

    @classmethod
    def set_client_socket(cls, client_socket):
        """
        クライアントのソケットをクラス変数にセット（シングルトン化）。
        """
        with cls._lock:
            if cls._client_socket is None:  # すでに設定されている場合は変更しない
                cls._client_socket = client_socket

                # 既存の `action_manager` を更新する
                if cls._instance.action_manager:
                    cls._instance.action_manager.client_socket = client_socket
                else:
                    cls._instance.action_manager = ActionManager(client_socket)

                print(f"✅ `client_socket` が設定されました: {client_socket}")

    def initialize_agent(self):
        """
        LangChain の Agent をセットアップし、適切なアクションを実行できるようにする。
        """
        tools = [
            Tool(name="Speak", func=self._instance.action_manager.speak, description="会話をする"),
            Tool(name="SearchMemory", func=self._instance.action_manager.search_memory, description="記憶を検索する"),
            Tool(name="AddMemory", func=self._instance.action_manager.add_memory, description="新しい記憶を保存する"),
            Tool(name="PromoteToLTM", func=self._instance.action_manager.promote_to_ltm, description="重要な記憶を LTM に移動する"),
        ]

        return initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=BedrockChat(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"),
            verbose=True,
        )

    def decide_action(self, user_input=None):
        """
        Internal State（VAE の潜在変数）を読み取り、意思決定を行う。デフォルトでは何もしない。
        """
        self.internal_state.update_all_states()
        current_state = self.internal_state.get_internal_state()

        threshold = 0.5  # しきい値（調整可能）

        # 総合差分スコアを計算
        difference = np.array(current_state) - self.ideal_state
        total_diff = np.linalg.norm(difference)

        # Agent に渡すプロンプト
        prompt = f"""
        あなたの目標は、「内部状態を快に近づけること」です。
        そのために以下のことを意識してください。

        - 内部状態の改善を目指して戦略を立ててください。
        - 実行可能なアクションツール（例: Speak, AddMemory, Learn）を活用してください。
        - 自身の発話や行動によって、相手からの反応や記憶内容を学習してください。

        【現在の内部状態の差分】
        {difference.tolist()}

        【総合変化量スコア】（しきい値: {threshold}）
        {total_diff}

        【ルール】
        - 変化量が {threshold} 未満の場合、何もせず静かに待つ。
        - 変化量が {threshold} 以上の場合、適切なアクションを決定する。
        - 必要な場合は、過去の会話履歴を参照して自然な会話を行う。

        【期待する出力】
        - `Speak("返答内容")`
        - `SearchMemory("検索クエリ")`
        - `AddMemory("記憶する内容")`
        """

        if total_diff >= threshold:
            response = self.agent.run(prompt)
            self.action_manager.speak(response)


    def autonomous_behavior(self):
        """彼女AIが自発的に Internal State を分析し、適切なアクションを実行"""
        while True:
            time.sleep(random.randint(10, 20))
            self.decide_action()

