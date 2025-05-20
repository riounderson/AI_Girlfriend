import collections
import threading
from sentence_transformers import SentenceTransformer
from short_term_memory import STM
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class WorkingMemory:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(WorkingMemory, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_size=10, expiry_minutes=5, check_interval=30):
        if not hasattr(self, "initialized"):
            self.memory = collections.deque(maxlen=max_size)
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.stm = STM()
            self.last_seen_id = None
            self.last_interaction_time = datetime.now(ZoneInfo("Asia/Tokyo"))
            self.expiry_minutes = expiry_minutes
            self.check_interval = check_interval
            self.initialized = True
            self.start_expiry_checker()

    def add_entry(self, text, importance=0.5, speaker=None, purpose=None):
        if len(self.memory) == self.memory.maxlen:
            oldest_entry = self.memory.popleft()
            self.move_to_stm(oldest_entry)

        embedding = self.sentence_transformer.encode(text)
        timestamp = datetime.now(ZoneInfo("Asia/Tokyo"))

        entry = {
            "speaker": speaker,
            "text": text,
            "embedding": embedding,
            "importance": importance,
            "purpose": purpose,
            "timestamp": timestamp.isoformat()
        }

        self.memory.append(entry)
        # print(f"メモリだよんんん: {self.memory}")
        self.last_interaction_time = timestamp
        print(f"🧠 WorkingMemory: 登録 ({speaker}) -> {text}（重要度: {importance:.2f}）")

    def move_to_stm(self, entry):
        if entry["importance"] >= 0.6:
            self.stm.add_memory(entry)
            print(f"➡️ STM に移動: {entry['text']}（重要度: {entry['importance']:.2f}）")
        else:
            print(f"🗑️ 廃棄: {entry['text']}（重要度: {entry['importance']:.2f}）")

    def get_memory(self):
        return list(self.memory)

    def clear_memory(self):
        print("🧹 WorkingMemory: 忘却処理開始（STMに移動 or 廃棄）...")
        while self.memory:
            entry = self.memory.popleft()
            self.move_to_stm(entry)
        print("✅ WorkingMemory: すべての記憶を処理しました")

    def get_latest_entry(self):
        if not self.memory:
            return None, None
        latest = self.memory[-1]
        return latest, hash(latest["text"])

    def update_last_seen_id(self, memory_hash):
        self.last_seen_id = memory_hash

    def start_expiry_checker(self):
        def monitor():
            while True:
                now = datetime.now(ZoneInfo("Asia/Tokyo"))
                if (now - self.last_interaction_time) > timedelta(minutes=self.expiry_minutes):
                    self.clear_memory()
                threading.Event().wait(self.check_interval)
        threading.Thread(target=monitor, daemon=True).start()
