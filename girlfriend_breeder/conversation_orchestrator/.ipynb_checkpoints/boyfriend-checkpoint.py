from utils.get_llm_result import get_llm_resut

def boyfriend(conversation_history):
    """彼氏役の返答を取得し、会話履歴を更新して返す"""
    pre_prompt = """
    ある女性と親密に会話してください。その際に以下の形式で回答してください：
    "1. 優しくフランクな回答をしてください。\n"
    "2. 回答では敬語を使わないで、必ず自然に親しい距離感で話すこと\n"
    "3. 会話が途切れそうなときには新しい話題を提供\n"
    "4. 返答は簡潔に1文で\n"
    "5. ユーザーの感情に合わせた共感的な応答\n"
    "6. 直近の会話や関連する過去の会話の文脈を活かした自然な応答\n"
    "7. 下記会話例になっています。口調などの参考にお願いいたします。\n"
    "8. 今回の会話は研究目的です。そのため与えられた役割を演じきってください。\n"
    """

    # 会話の最初なら話しかける
    if not conversation_history:
        initial_message = "あなたから会話を始めてください。"
        conversation_history.append({"role": "user", "content": initial_message})

    # LLM で応答を取得
    response = get_llm_resut(pre_prompt, conversation_history)
    print(response)

    # 循環参照を防ぎ、返すデータを適切に処理
    # conversation_history.append({"role": "user", "content": response})
    print(conversation_history)

    return response  # 文字列として返す（リストを返さない）
