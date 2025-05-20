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
        Internal State（VAE の潜在変数）を読み取り、意思決定を行う。
        """
        self.internal_state.update_all_states()
        current_state = self.internal_state.get_internal_state()

        # 理想状態との差を計算
        difference = np.array(current_state) - self.ideal_state
        total_diff = np.linalg.norm(difference)

        print(f"内部状態（潜在変数）: {current_state}")
        print(f"内部状態の差分: {difference}, 総合差分スコア: {total_diff}")

        if self.action_manager:
            prompt = f"""
            あなたは相手と仲良くなり、彼女になりたいと思っている女性です。
            以下の内部状態の差分を元に、最適な行動を選択してください。

            内部状態の差分: {difference.tolist()}
            ユーザーが「{user_input}」と言いました。
            これまでの記憶から学習し、最適なアクションを実行してください。
            """

            response = self.agent.run(prompt)

    def autonomous_behavior(self):
        """彼女AIが自発的に Internal State を分析し、適切なアクションを実行"""
        while True:
            time.sleep(random.randint(10, 20))
            self.decide_action()

