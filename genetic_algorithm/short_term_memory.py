from Utils.manage_memory import ChromaMemory
from long_term_memory import LTM 
import datetime

class STM(ChromaMemory):
    """
    短期記憶 (STM) を管理するクラス
    - 通常の短期記憶（会話や記憶）
    - 学習成果（報酬付きの戦略履歴）
    """
    def __init__(self, db_path="memory-db/STM", collection_name="STM_memory", max_size=20):
        super().__init__(db_path, collection_name)
        self.max_size = max_size
        self.ltm = LTM()

        # ⬇️ 学習成果記憶用リスト
        self.learning_results = []

    def add_short_term_memory(self, text, metadata=None):
        """
        通常の STM に会話や記憶を追加（最大数制限あり）
        """
        if len(self.collection.get()["documents"]) >= self.max_size:
            self.delete_oldest_memory()

        memory_id = self.add_memory(text, metadata)
        print(f"📝 短期記憶に追加: {text} (ID: {memory_id})")
        return memory_id

    def delete_oldest_memory(self):
        """
        最も古い記憶を削除（FIFO）
        """
        records = self.collection.get()
        if records["documents"]:
            oldest_id = records["ids"][0]
            self.delete_memory(oldest_id)
            print(f"🗑️ 最も古い記憶を削除: {records['documents'][0]} (ID: {oldest_id})")

    def promote_to_ltm(self, threshold=2):
        """
        頻出する記憶を LTM に昇格
        """
        records = self.collection.get()
        frequency_map = {}

        for doc in records["documents"]:
            frequency_map[doc] = frequency_map.get(doc, 0) + 1

        for text, count in frequency_map.items():
            if count >= threshold:
                self.ltm.add_long_term_memory(text)
                self.delete_memory(str(hash(text)))
                print(f"🚀 記憶を LTM に昇格: {text}（出現回数: {count}）")

    def add_learning_result(self, purpose: str, feedback: str, reward: float):
        """
        短期的な学習結果（目的・会話・報酬）を記録
        """
        result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "purpose": purpose,
            "feedback": feedback,
            "reward": reward
        }
        self.learning_results.append(result)
        print(f"📚 学習成果を STM に記録: {purpose} / 報酬: {reward:.2f}")

    def get_learning_memory(self):
        """保存された学習記憶を取得"""
        return self.learning_results
