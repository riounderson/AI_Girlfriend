import threading
import time
from cognitive_controller import CognitiveController
from actions import ActionManager
from Modules.sociality_module import SocialityModule
from Modules.common_sense_module import CommonSenseModule
from Modules.reward_module import RewardModule
from Modules.goal_module import GoalModule
from Modules.recognition_module import RecognitionModule
from Modules.evaluate_action_module import ShortRewardModule


class Girlfriend:
    """
    彼女AIの全体を管理するクラス
    - 各モジュール (欲求・社会性など) を並列に動かす
    - 彼氏の発話を聞く (聴覚)
    - 内部状態を読んで意思決定する (意識)
    """

    def __init__(self, client_socket=None):
        print(f"Girlfriendモジュール内のソケット確認!!!!: {client_socket}")
        # === モジュール群のインスタンス化 ===
        self.client_socket = client_socket
        self.sociality_module = SocialityModule()
        self.common_sense_module = CommonSenseModule()
        self.reward_module = RewardModule()
        self.goal_module = GoalModule()
        self.recognition_module = RecognitionModule()
        self.evaluate_action_module = ShortRewardModule()

        # === 意識と感覚 ===
        self.cognitive_controller = CognitiveController()
        print(f"彼女の中でソケットを再度確認！: {self.client_socket}")
        self.am = ActionManager()
        self.am.update_client_socket(self.client_socket)
        # self.action_manager = ActionManager(self.client_socket)

        # === モジュールリスト (関数, 間隔[秒]) ===
        self.modules = [
            (self.sociality_module.calculate_sociality, 20),
            (self.common_sense_module.evaluate_common_sense, 20),
            (self.reward_module.evaluate_longtime_reward, 90),
            (self.goal_module.update_internal_goal, 300),
            (self.recognition_module.calculate_recognition, 300),
            (self.evaluate_action_module.evaluate_shortterm_reward, 180),
        ]

    def run_module_loop(self, func, interval):
        """個別モジュールを指定間隔で永続ループ実行する"""
        while True:
            try:
                func()
            except Exception as e:
                print(f"⚠️ {func.__name__} の実行中にエラー発生: {e}")
            time.sleep(interval)

    def start_modules(self):
        """各モジュールを並列実行"""
        for func, interval in self.modules:
            thread = threading.Thread(target=self.run_module_loop, args=(func, interval), daemon=False)
            thread.start()

    def start_cognitive_controller(self):
        """内部状態をもとに行動を決める意識の起動"""
        thread = threading.Thread(target=self.cognitive_controller.run, daemon=False)
        thread.start()

    def start_listen(self):
        """彼氏の発話を受け取る耳 (リスナー) を起動"""
        thread = threading.Thread(target=self.am.listen, daemon=False)
        thread.start()

    def start_all(self):
        """彼女AI全体を起動"""
        self.start_modules()
        self.start_cognitive_controller()
        self.start_listen()
