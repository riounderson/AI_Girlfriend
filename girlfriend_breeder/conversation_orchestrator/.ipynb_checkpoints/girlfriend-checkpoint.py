from utils.get_llm_result import get_llm_resut

def girlfriend(gf_system_prompt, conversation_history):
    """彼女役の返答を取得し、会話履歴を更新して返す"""
    pre_prompt = gf_system_prompt

    # LLM で応答を取得
    response = get_llm_resut(pre_prompt, conversation_history)
    print(f"彼女の応答♡: {response}")

    # 循環参照を防ぎ、リストを適切に更新
    # conversation_history.append({"role": "user", "content": response})
    # print(conversation_history)

    return response  # 文字列として返す
