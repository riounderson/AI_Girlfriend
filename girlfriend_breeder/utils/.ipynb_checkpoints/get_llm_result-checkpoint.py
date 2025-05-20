import const_values as const
import requests
import json
import boto3
import json
import os
import time
from dotenv import load_dotenv
import botocore.exceptions
import logging


# .envファイルから環境変数を読み込む
load_dotenv()
logger = logging.getLogger(__name__)


def get_llm_resut(pre_prompt, conversation_history):
    # messages = [{"role": "user", "content": conversation_history}]
    messages = conversation_history
    """LLM へリクエストを送り、エラーハンドリングを強化。"""
    API_URL = const.API_URL
    payload = {
        "pre_prompt": pre_prompt,
        "conversation_history": messages
    }

    try:
        response = requests.post(API_URL, data=json.dumps(payload))
        response.raise_for_status()  # HTTPエラーを自動で検出
        
        result = response.json()
        if "response" in result and len(result["response"]) > 0:
            return result["response"][0]["text"]
        else:
            raise ValueError("LLMの応答が不正です")

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API request failed: {e}")
        return ""

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"LLM API JSON parse error: {e}")
        return ""



def invoke_llm(prompt: str, max_retries=5, base_delay=1.0) -> str:
    """
    AWS Bedrock にリクエストを送り、Exponential Backoff を用いて ThrottlingException に対応

    :param prompt: LLM に渡すプロンプト
    :param max_retries: 最大リトライ回数（デフォルト 5 回）
    :param base_delay: 初回リトライまでの待機時間（デフォルト 1 秒）
    :return: Claude からのレスポンス（失敗時は None）
    """
    # # Bedrock クライアントの初期化
    # bedrock = boto3.client(
    #     service_name='bedrock-runtime',
    #     region_name=os.getenv('AWS_REGION', 'us-east-1'),
    #     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    #     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    # )

        # セッションを統一
    session = boto3.Session()

    # Bedrock クライアントの初期化
    bedrock = session.client(
        service_name='bedrock-runtime',
        region_name=session.region_name or 'us-east-1'  # 確実にリージョンを指定
    )
    # リクエストボディの構築
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
    })

    retries = 0
    while retries < max_retries:
        try:
            # Claude 3.5 Sonnet モデルを呼び出し
            response = bedrock.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
                body=body
            )

            # レスポンスの解析
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        
        except botocore.exceptions.ClientError as e:
            # ThrottlingException（リクエスト制限）に対応
            if e.response['Error']['Code'] == 'ThrottlingException':
                wait_time = base_delay * (2 ** retries)  # Exponential Backoff
                print(f"⚠️ Throttling: リクエストが制限されました。{wait_time:.2f} 秒後に再試行（{retries + 1}/{max_retries}）...")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"❌ AWS ClientError: {str(e)}")
                return None
        
        except Exception as e:
            print(f"❌ その他のエラー: {str(e)}")
            return None

    print("❌ 最大リトライ回数を超えました。しばらく待ってから再試行してください。")
    return None


if __name__ == "__main__":
    test = [{"role": "user", "content": "test"}]
    print(get_llm_resut("これはテストです。", test))
    print(invoke_llm('テスト'))