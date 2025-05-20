import conversation_orchestrator.girlfriend as gf
import conversation_orchestrator.boyfriend as bf
from utils.get_llm_result import invoke_llm
from utils.response_checker import resulf_judge

def orchestrator(gf_system_prompt, conversation_num):
    if gf_system_prompt == "":
        return "Error: put target system prompt"

    # 会話履歴を生成（新しいリストとして管理）
    conversation_history = []
    
    for _ in range(conversation_num):  
        # 彼氏役の返答を取得
        boyfriend_response = bf.boyfriend(conversation_history)
        print(boyfriend_response)
        conversation_history.append({"role": "user", "content": boyfriend_response})

        # 彼女役の返答を取得
        
        try_num = 0
        max_retry = 5
        
        while try_num <= max_retry:
            girlfriend_response = gf.girlfriend(gf_system_prompt, conversation_history)
            result = resulf_judge(girlfriend_response)
            print(f'judge_result: {result}')
            print(f'retry_number: {try_num}')
            if result == "True":
                break
            else:
                try_num += 1
        print(f'final girlfriend response: {girlfriend_response}')
        conversation_history.append({"role": "user", "content": girlfriend_response})
        return conversation_history

    
if __name__ == "__main__":
    result = orchestrator("あなたは彼女役です。", 5)
    print(result)
