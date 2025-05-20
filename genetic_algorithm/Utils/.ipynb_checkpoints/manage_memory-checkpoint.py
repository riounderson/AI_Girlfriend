import os
import fitz  # PyMuPDF
import chromadb
import pprint
import datetime

class ChromaMemory:
    def __init__(self, db_path="../memory-db/chroma_db", collection_name="memory"):
        """
        ChromaDB のセットアップ（保存場所を指定）
        :param db_path: ChromaDB のデータを保存するディレクトリ
        :param collection_name: ChromaDB のコレクション名
        """
        # ディレクトリが存在しない場合は作成
        os.makedirs(db_path, exist_ok=True)
        print(db_path)

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_memory(self, text, metadata=None, memory_id=None):
        """
        記憶を ChromaDB に追加する
        :param text: 記憶する文章
        :param metadata: 記憶のメタデータ（カテゴリなど）
        :param memory_id: 一意の ID（省略可能）
        """
        if memory_id is None:
            memory_id = str(hash(text))

        self.collection.upsert(
            documents=[text],
            ids=[memory_id]
        )
        return memory_id

    def search_memory(self, query_text, top_k=3):
        """
        類似した記憶を検索
        :param query_text: 検索する文章
        :param top_k: 取得する件数
        :return: 類似する記憶
        """
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        # 検索結果を整形して返す
        return {
            "texts": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def delete_memory(self, memory_id):
        """
        特定の記憶を削除
        :param memory_id: 削除する ID
        """
        self.collection.delete(ids=[memory_id])

    def extract_text_from_pdf(self, pdf_path):
        """
        PDF からテキストを抽出
        :param pdf_path: PDF のファイルパス
        :return: 抽出したテキスト
        """
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text

    def add_pdf_to_memory(self, pdf_path):
        """
        PDF を ChromaDB に保存
        :param pdf_path: PDF のファイルパス
        """
        text = self.extract_text_from_pdf(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        memory_id = str(hash(pdf_name))

        self.collection.add(
            documents=[text],
            metadatas=[{"filename": pdf_name}],
            ids=[memory_id]
        )
        return memory_id

if __name__ == "__main__":
    memory_db = ChromaMemory()
    
    # テストデータの追加（メタデータなし）
    texts = [
        "私の好きなカレー屋さんは渋谷にあります",
        "今日は晴れています",
        "プログラミングが楽しいです"
    ]
    
    for text in texts:
        memory_db.add_memory(text)  # メタデータなしで追加
    
    # 検索テスト
    query_word = input("検索ワードを入力してください: ")
    result = memory_db.search_memory(query_word)
    
    if result["texts"]:
        print("\n検索結果:")
        for text, distance in zip(result["texts"], result["distances"]):
            print(f"テキスト: {text}")
            print(f"類似度スコア: {distance}\n")
    else:
        print("該当する結果が見つかりませんでした")
