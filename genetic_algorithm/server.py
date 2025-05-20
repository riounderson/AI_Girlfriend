import time
import random
import threading
import socket
from cognitive_controller import CognitiveController
from actions import ActionManager  
from girlfriend import Girlfriend
import multiprocessing as mp
import faulthandler, sys
faulthandler.enable(file=sys.stderr, all_threads=True)



HOST = "127.0.0.1"
PORT = 65432

class AI_Server:
    def __init__(self):
        self.controller = CognitiveController()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(1)
        print(f"👩 彼女AI: サーバー起動中... ポート {PORT} で待機中")

    def run(self):
        """ユーザーの発話を待機しつつ、彼女AIの自発的発話を並行実行"""
        client_socket, addr = self.server_socket.accept()
        print(f"🖥️ クライアントが接続しました: {addr}")
        print(f"Serverモジュール内のソケット確認!!!!: {client_socket}")

        girlfriend = Girlfriend(client_socket=client_socket)
        girlfriend.start_all()

if __name__ == "__main__":
    faulthandler.enable(file=sys.stderr, all_threads=True)
    server = AI_Server()
    server.run()
    print("🟢 サーバー／Girlfriend 全スレッド起動完了。Ctrl+C で停止します。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 サーバー停止シグナルを受け取りました。シャットダウンします。")