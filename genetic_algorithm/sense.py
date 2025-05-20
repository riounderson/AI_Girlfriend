import socket
import threading

class Sense:
    """
    彼女AIの「耳」として機能し、彼氏（クライアント）からの発話を受け取るクラス
    """
    def __init__(self, client_socket, cognitive_controller):
        """
        :param client_socket: クライアント（彼氏）との通信を行うソケット
        :param cognitive_controller: 彼女AIの意識を司る CognitiveController インスタンス
        """
        self.client_socket = client_socket
        self.cognitive_controller = cognitive_controller
        self.running = True  # ループ制御用フラグ

    def listen(self):
        """
        クライアント（彼氏）からの発話を受け取り、CognitiveController に渡す
        """
        print("👂 彼女AIの耳（Sense）が起動しました。")

        while self.running:
            try:
                user_input = self.client_socket.recv(1024).decode("utf-8").strip()
                if not user_input or user_input.lower() == "exit":
                    print("🖥️ クライアントが切断されました。")
                    self.running = False
                    break

                print(f"🧑 あなた: {user_input}")

                # CognitiveController に発話を渡して処理
                self.cognitive_controller.process_input(user_input)

            except ConnectionResetError:
                print("⚠️ クライアント接続がリセットされました。")
                self.running = False
                break
            except UnicodeDecodeError:
                print("⚠️ 受信データのデコードに失敗しました。")
                continue  # デコードエラーが発生してもループを続行

        self.client_socket.close()
