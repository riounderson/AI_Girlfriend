from Utils.manage_memory import ChromaMemory

class LTM(ChromaMemory):
    """
    長期記憶 (LTM) を管理するクラス
    """
    def __init__(self, db_path="memory-db/LTM", collection_name="LTM_memory"):
        super().__init__(db_path, collection_name)

    def add_long_term_memory(self, text, metadata=None):
        """
        重要な記憶を LTM に追加
        """
        memory_id = self.add_memory(text, metadata)
        print(f"🚀 長期記憶に追加: {text} (ID: {memory_id})")
        return memory_id

    def get_all_memories(self):
        """
        LTM に保存された全ての記憶を取得
        """
        return self.collection.get()

    def forget_memory(self, memory_id):
        """
        LTM から特定の記憶を削除（古くなったものを削除するなど）
        """
        self.delete_memory(memory_id)
        print(f"🗑️ 長期記憶から削除: ID={memory_id}")
