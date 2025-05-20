from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.chat_models import BedrockChat
from Utils.internal_state import InternalState
from actions import ActionManager
import numpy as np

class CognitiveController:
    """
    彼女AIの自我を司るクラス。
    Internal State を VAE の潜在変数として管理し、意思決定に活用する。
    """
    def __init__(self):
        self.internal_state = InternalState()
        self.action_manager = ActionManager()
        self.ideal_state = np.array([0.8, 0.8, 0.8, 0.8])  # 理想の内部状態
        self.feature_names = ["mood", "affinity", "interest", "conversation"]

        # LangChain の Agent をセットアップ
        self.agent = self.initialize_agent()

    def initialize_agent(self):
        """
        LangChain の Agent をセットアップし、適切なアクションを実行できるようにする。
        """
        tools = [
            Tool(name="Speak", func=self.action_manager.speak, description="会話をする"),
            Tool(name="SearchMemory", func=self.action_manager.search_memory, description="記憶を検索する"),
            Tool(name="AddMemory", func=self.action_manager.add_memory, description="新しい記憶を保存する"),
            Tool(name="PromoteToLTM", func=self.action_manager.promote_to_ltm, description="重要な記憶を LTM に移動する"),
        ]

        return initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=BedrockChat(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"),
            verbose=True,
        )

    def decide_action(self, user_input):
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

        # どの要素がズレているかを解釈
        differences_text = []
        for i, diff in enumerate(difference):
            if diff < -0.2:  
                differences_text.append(f"{self.feature_names[i]} が低すぎる")
            elif diff > 0.2:
                differences_text.append(f"{self.feature_names[i]} が高すぎる")

        print(f"🔍 差分の解釈: {differences_text}")

        if not differences_text:
            print("🟢 内部状態が理想に近いため、行動の必要なし。")
            return

        # LangChain Agent に差分を解釈させ、適切なアクションを選択
        prompt = f"""
        あなたは相手と仲良くなり、彼女になりたいと思っている女性です。以下の内部状態の差分を元に、最適な行動を選択してください。

        内部状態の差分: {differences_text}
        ユーザーが「{user_input}」と言いました。
        これまでの記憶から学習し、最適なアクションを実行してください。
        """

        self.agent.run(prompt)

    def chat(self):
        """
        ユーザーが終了するまで会話を続ける
        """
        print("👩 彼女AI: こんにちは！何か話したいことはある？（終了するには 'exit' を入力）")

        while True:
            user_input = input("🧑 あなた: ")
            if user_input.lower() == "exit":
                print("👩 彼女AI: じゃあね！また話そうね！")
                break

            self.decide_action(user_input)

if __name__ == "__main__":
    controller = CognitiveController()
    controller.chat()
