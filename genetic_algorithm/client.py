import socket
import threading
import sys

HOST = "127.0.0.1"
PORT = 65432

class AI_Client:
    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((HOST, PORT))
        print("🧑 クライアント: 彼女AIと接続しました。話しかけてください！（'exit' で終了）")

        # 受信スレッドを開始
        self.running = True
        self.receive_thread = threading.Thread(target=self.receive_messages, daemon=True)
        self.receive_thread.start()

    def receive_messages(self):
        """彼女AIからのメッセージをリアルタイムに受信して表示"""
        while self.running:
            try:
                ai_response = self.client_socket.recv(1024).decode("utf-8", errors="ignore")
                if ai_response:
                    print(f"\n👩 彼女AI: {ai_response}\n🧑 あなた: ", end="", flush=True)  # 次の入力のプロンプトを保持
            except UnicodeDecodeError as e:
                print(f"⚠️ デコードエラー: {e}")
            except ConnectionResetError:
                print("⚠️ サーバーとの接続が切断されました。")
                self.running = False
                break
            except Exception as e:
                print(f"⚠️ クライアント側エラー: {e}")
                break

    def chat(self):
        """ユーザーが彼女AIに話しかけ、AI の応答を受信"""
        while True:
            user_input = input("🧑 あなた: ")
            self.client_socket.sendall(user_input.encode("utf-8"))
            if user_input.lower() == "exit":
                self.running = False  # 受信スレッドを停止
                break

        self.client_socket.close()

if __name__ == "__main__":
    print(f"入力エンコーディング: {sys.stdin.encoding}")
    client = AI_Client()
    client.chat()
