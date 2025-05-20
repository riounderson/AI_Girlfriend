import datetime
import socket
import threading
from long_term_memory import LTM
from short_term_memory import STM 
from Modules.working_memory import WorkingMemory
from Utils.invoke_llm import get_claude_response

class ActionManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("🆕 ActionManager 初回インスタンス生成")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, client_socket=None):
        if not hasattr(self, 'initialized'):
            self.ltm = LTM()
            self.stm = STM()
            self.wm = WorkingMemory(max_size=10)
            self.client_socket = client_socket
            self.initialized = True  # 再初期化防止

    def update_client_socket(self, new_socket):
        print(f"🔁 クライアントソケットを更新します: {new_socket}")
        self.client_socket = new_socket

    def listen(self):
        """ユーザーの発話を受け取り、重要度評価し、AI 応答とともに会話ターンとして WM に記録"""
        print("👂 彼女AIの耳（リスナー）起動中")

        while True:
            try:
                user_input = self.client_socket.recv(1024).decode("utf-8").strip()
                if not user_input or user_input.lower() == "exit":
                    print("🖥️ クライアントが切断されました。")
                    break

                print(f"🧑 あなた: {user_input}")
                speaker = "others"

                # ① LLM に重要度を評価させる
                importance = self.evaluate_importance(user_input)
                # importance = 0.1
                print(f"importanceでござる: {importance}")

                # ③ 会話ターンを WM に記録
                self.wm.add_entry(text=user_input, importance=importance, speaker=speaker)

            except ConnectionResetError:
                print("⚠️ クライアント接続がリセットされました。")
                break
            except UnicodeDecodeError:
                print("⚠️ 受信データのデコードに失敗しました。")
                continue  # デコードエラーが発生してもループを継続
        # self.client_socket.close()


    def evaluate_importance(self, text):
        """LLM に発話の重要度を評価させる（0~1）"""
        prompt = f"""
        以下のユーザーの発話の重要度を 0 から 1 のスコアで評価してください。
        重要度は’あなたにとってその人の発話した内容が重要であるかどうか'で判断してください。
        例えば、関係値の低い相手に好意を伝えられてもどうでもいいと思いますが、関係値の高い相手もしくは好意を持っている相手に言われると重要だと思います。
        **なお評価の理由など余計な情報は一切いりません。点数だけを出力してください。
        - 0 に近い → 重要ではない
        - 1 に近い → 非常に重要

        ユーザーの発話: "{text}"

        【回答例】
        0.8
        """
        print(f"ここまで来てるんかい？？！")
        print(f"Who are you??!: {prompt}")
        response = get_claude_response(prompt)

        print(f"response!!!! {response}")

        # 文字列を数値化（エラー時は 0.5）
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # 0~1 の範囲に制限
        except ValueError:
            return 0.5  # デフォルトのスコア

    def move_wm_to_stm(self):
        """
        WM のデータを STM に移動し、古いデータを削除
        """
        oldest_entry = self.wm.memory.popleft()  # FIFO で最も古いデータを取得
        if oldest_entry["importance"] >= 0.6:
            self.stm.add_short_term_memory(oldest_entry["text"])
            print(f"🚀 STM に移動: {oldest_entry['text']}（重要度: {oldest_entry['importance']:.2f}）")
        else:
            print(f"🗑️ {oldest_entry['text']} は STM には移動せず、削除されました。")

    def promote_memory(self):
        """
        STM → LTM に記憶を昇格させる
        """
        self.stm.promote_to_ltm(threshold=2)

    def search_memory(self, query, memory_type="STM", top_k=3):
        """
        記憶を検索し、最も関連性の高い情報を取得
        :param query: 検索するキーワード
        :param memory_type: "STM"（短期記憶）または "LTM"（長期記憶）
        :param top_k: 取得する件数
        """
        memory_db = self.stm if memory_type == "STM" else self.ltm
        result = memory_db.search_memory(query, top_k=top_k)

        if result["texts"]:
            print(f"🔍 {memory_type} の検索結果:")
            for text, distance in zip(result["texts"], result["distances"]):
                print(f"📌 記憶: {text}（類似度: {distance:.2f}）")
            return result["texts"][0]  # 最も関連性の高いものを返す
        else:
            print(f"⚠️ {memory_type} に該当する記憶が見つかりませんでした")
            return None

    def speak(self, text, purpose=None):
        """彼女AIが発話する（クライアントに送信）"""
        print(f"👩 彼女AI: {text}")  # サーバー側（AI）にも表示
        
        if self.client_socket and isinstance(self.client_socket, socket.socket):
            try:
                speaker = "self"
                self.wm.add_entry(text=text, speaker=speaker, purpose=purpose)
                self.client_socket.sendall(text.encode("utf-8"))
                print("✅ クライアントへの送信成功")  # デバッグ用
            except Exception as e:
                print(f"⚠️ クライアントへの送信失敗: {e}")
        else:
            print("⚠️ `client_socket` が無効です。")
