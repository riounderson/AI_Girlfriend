# =========================================
# invoke_llm.py  (丸ごと置き換え例)
# =========================================
import os, json, time, threading, inspect
from queue import Queue
import boto3, botocore
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# ---------- Bedrock 定数 ----------
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
REGION    = os.getenv("AWS_REGION", "us-east-1")
AWS_KEY   = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET= os.getenv("AWS_SECRET_ACCESS_KEY")

# ---------- 共通設定 ----------
_bedrock_cfg = Config(
    read_timeout=60,
    connect_timeout=20,
    max_pool_connections=10,
)

# ---------- 内部ヘルパ ----------
def _build_body(prompt: str) -> str:
    """Bedrock invoke_model 用 JSON ボディを構築"""
    return json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    })

# ---------- Queue & Worker ----------
_Request = tuple[str, threading.Event, dict]   # (prompt, event, holder)
_req_q: Queue[_Request] = Queue()

_client = None
def get_bedrock_client():
    global _client
    if _client is None:
        # プロセスID情報をログに出力
        import os
        pid = os.getpid()
        print(f"🆔 プロセス {pid} で Bedrock クライアント初期化")
        
        # クライアント設定をより堅牢に
        _client = boto3.client(
            "bedrock-runtime",
            region_name=REGION,
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
            config=Config(
                read_timeout=15,
                connect_timeout=5,
                max_pool_connections=5,  # プロセスごとに少なく
                retries={"max_attempts": 1}
            )
        )
    return _client

def _llm_worker() -> None:
    """Bedrock 呼び出し専用スレッド（常駐）"""
    print("🛠️  LLMワーカー起動: Bedrock クライアント初期化")
    try:
        client = get_bedrock_client()
        
        while True:
            try:
                # キューが空なら少し待つ
                if _req_q.empty():
                    time.sleep(0.1)
                    continue
                
                prompt, ev, holder = _req_q.get(block=True, timeout=1.0)
                print(f"📤 リクエスト処理開始: {prompt[:30]}...")
                
                # 別スレッドでAPI呼び出しを行い、タイムアウト制御
                result_dict = {"completed": False}
                api_thread = threading.Thread(
                    target=invoke_with_timeout,
                    args=(prompt, result_dict, client)
                )
                api_thread.daemon = True
                api_thread.start()
                
                # 明示的なタイムアウト（15秒）
                api_thread.join(15)
                
                if not result_dict.get("completed", False):
                    print("⚠️ API呼び出しタイムアウト")
                    holder["error"] = TimeoutError("API call timed out")
                elif "error" in result_dict:
                    holder["error"] = result_dict["error"]
                    print(f"❌ API呼び出しエラー: {result_dict['error']}")
                else:
                    holder["result"] = result_dict["result"]
                    print("✅ API呼び出し成功")
                
                # 完了通知
                ev.set()
                _req_q.task_done()
                
            except Exception as e:
                import traceback
                print(f"💥 ワーカーループエラー: {e}")
                print(f"スタックトレース: {traceback.format_exc()}")
    
    except Exception as e:
        import traceback
        print(f"💥💥 ワーカー致命的エラー: {e}")
        print(f"スタックトレース: {traceback.format_exc()}")

def invoke_with_timeout(prompt, result_dict, client):
    try:
        # ...ここは既存コードと同じ...
        body = _build_body(prompt) 
        print(f"📡 invoke_model 送信開始")
        
        resp = client.invoke_model(modelId=MODEL_ID, body=body)
        print(f"✅ invoke_model 応答受信")
        
        raw = resp["body"].read()
        print(f"✅ body.read() 完了")
        
        result_dict["result"] = json.loads(raw)["content"][0]["text"]
        result_dict["completed"] = True
        
    except Exception as e:
        import traceback
        result_dict["error"] = e
        result_dict["stack"] = traceback.format_exc()
        result_dict["completed"] = True
        print(f"❌ API呼び出し例外: {type(e).__name__}: {e}")
        print(f"詳細: {traceback.format_exc()}")

# グローバル変数として保持
_worker_thread = None

def ensure_worker_running():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_llm_worker, daemon=False)
        _worker_thread.start()

# デーモン起動（メインが終われば自動終了）
worker_thread = threading.Thread(target=_llm_worker, daemon=False)
worker_thread.start()

# ---------- パブリック API ----------
def get_claude_response(prompt: str, timeout: float = 30) -> str:
    caller = inspect.stack()[1]
    print(f"🚀 Claudeリクエスト開始  ← {caller.function}()")

    # ★ ここを修正 ★
    ev: threading.Event = threading.Event()
    store: dict[str, object] = {}

    _req_q.put((prompt, ev, store))

    if not ev.wait(timeout):
        raise TimeoutError("Bedrock did not return within timeout")

    if "error" in store:
        raise store["error"]
    return store["result"]

# ---------- 動作確認 ----------
if __name__ == "__main__":
    prompt = "Python の主な特徴を 3 行で教えてください。"
    print("🎯 prompt:", prompt)
    try:
        print("🧠 Claude:", get_claude_response(prompt))
    except Exception as e:
        print("💥 エラー:", e)
