import time
import random
import threading
import socket
from cognitive_controller import CognitiveController
from actions import ActionManager
from Modules.affinity_module import AffinityModule
import numpy as np


HOST = "127.0.0.1"
PORT = 65432

class AI_Server:
    def __init__(self):
        self.controller = CognitiveController()
        self.affinity_module = AffinityModule()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(1)
        print(f"👩 彼女AI: サーバー起動中... ポート {PORT} で待機中")

    def listen(self, client_socket):
        """ユーザーの発話を受け取り、各モジュールが並行して評価し、Internal State に反映"""
        print(f"👂 彼女AIの耳（リスナー）起動: ポート {PORT} で待機中")
        # action_manager = ActionManager(client_socket)

        while True:
            try:
                user_input = client_socket.recv(1024).decode("utf-8").strip()
                if not user_input or user_input.lower() == "exit":
                    print("🖥️ クライアントが切断されました。")
                    break

                print(f"🧑 あなた: {user_input}")

                # 各モジュールの評価を並行処理で実行
                threads = []
                modules = [self.affinity_module]

                for module in modules:
                    thread = threading.Thread(target=module.calculate_affinity, args=(user_input,), daemon = True)
                    threads.append(thread)
                    thread.start()

                # すべてのモジュールの評価が終わるのを待つ
                for thread in threads:
                    thread.join()

            except ConnectionResetError:
                print("⚠️ クライアント接続がリセットされました。")
                break
            except UnicodeDecodeError:
                print("⚠️ 受信データのデコードに失敗しました。")
                continue  # デコードエラーが発生してもループを続行

        client_socket.close()



    def run(self):
        """ユーザーの発話を待機しつつ、彼女AIの自発的発話を並行実行"""
        client_socket, addr = self.server_socket.accept()
        print(f"🖥️ クライアントが接続しました: {addr}")

        cognitive_controller = CognitiveController()  # シングルトンのインスタンスを取得
        CognitiveController.set_client_socket(client_socket)  # クライアントソケットを設定

        thread1 = threading.Thread(target=self.listen, args=(client_socket,), daemon=True)
        thread2 = threading.Thread(target=cognitive_controller.autonomous_behavior, daemon=True)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

if __name__ == "__main__":
    server = AI_Server()
    server.run()
