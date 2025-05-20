import datetime
import socket
from Utils.manage_memory import ChromaMemory
from Utils.invoke_llm import get_claude_response

class ActionManager:
    """
    彼女AIが取ることのできるアクションを管理するクラス。
    記憶の追加、検索、発話などを制御する。
    """
    def __init__(self, client_socket=None):
        """LTM と STM の記憶管理を初期化"""
        self.ltm = ChromaMemory(db_path="memory-db/LTM", collection_name="LTM_memory")
        self.stm = ChromaMemory(db_path="memory-db/STM", collection_name="STM_memory")
        self.client_socket = client_socket

    def thinking_prompt(self, prompt="test-hello"):
        # prompt = 
        response = get_claude_response(prompt)
        return response

    def speak(self, text):
        """彼女AIが発話する（クライアントに送信）"""
        print(f"👩 彼女AI: {text}")  # サーバー側（AI）にも表示

        if self.client_socket and isinstance(self.client_socket, socket.socket):
            try:
                self.client_socket.sendall(text.encode("utf-8"))
                print("✅ クライアントへの送信成功")  # デバッグ用
            except Exception as e:
                print(f"⚠️ クライアントへの送信失敗: {e}")
        else:
            print("⚠️ `client_socket` が無効です。")


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

    def add_memory(self, text, memory_type="STM"):
        """
        記憶を追加（LTM または STM）
        :param text: 記憶するテキスト
        :param memory_type: "STM"（短期記憶）または "LTM"（長期記憶）
        """
        memory_db = self.stm if memory_type == "STM" else self.ltm
        memory_id = memory_db.add_memory(text, metadata={"timestamp": datetime.datetime.now().isoformat()})
        print(f"✅ {memory_type} に記憶を追加: {text} (ID: {memory_id})")

    def promote_to_ltm(self, memory_id):
        """
        STM の記憶を LTM に昇格
        :param memory_id: STM の記憶ID
        """
        result = self.stm.collection.get(ids=[memory_id])
        if result["documents"]:
            text = result["documents"][0]
            metadata = result["metadatas"][0]
            self.ltm.add_memory(text, metadata, memory_id)
            self.stm.delete_memory(memory_id)
            print(f"🚀 記憶 {memory_id} を STM → LTM に昇格しました！")
        else:
            print(f"⚠️ 記憶 {memory_id} が STM に存在しません")

if __name__ == "__main__":
    action_manager = ActionManager()
    
    # 彼女AIが発話
    print(action_manager.thinking_prompt("test"))

    # # 記憶を追加
    # action_manager.add_memory("今日はカレーを食べました", memory_type="STM")

    # # 記憶を検索
    # query = input("🔍 検索ワードを入力してください: ")
    # retrieved_memory = action_manager.search_memory(query, memory_type="STM")

    # # 重要な記憶なら LTM に昇格
    # if retrieved_memory:
    #     promote = input("📌 この記憶を LTM に保存しますか？（yes/no）: ")
    #     if promote.lower() == "yes":
    #         memory_id = str(hash(retrieved_memory))
    #         action_manager.promote_to_ltm(memory_id)

