import logging
import time
from typing import List, Dict
from conversation_orchestrator.orchestrator import orchestrator
from utils.get_llm_result import invoke_llm
from modul_types import Population, EvolutionUnit

logger = logging.getLogger(__name__)

def evaluate_population(population: Population, num_evals: int) -> Population:
    """ Evaluates the effectiveness of each system prompt by running a conversation test and scoring it.

    Args:
        population (Population): The evolving population.
        num_evals (int): Number of evaluation examples.

    Returns:
        Population: Population with updated fitness scores.
    """
    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    elite_fitness = -1

    for unit in population.units:
        unit.fitness = 0  # 適応度のリセット

        # 進化したシステムプロンプト (`unit.P`) を `orchestrator` に渡して会話履歴を生成
        conversation_history = orchestrator(unit.P, num_evals)

        # 会話履歴を LLM で評価し、スコアリング
        evaluation_scores = evaluate_conversation(conversation_history)

        # JSON の解析に失敗した場合、デフォルト値を設定
        if not evaluation_scores or not isinstance(evaluation_scores, dict):
            logger.error(f"Invalid evaluation scores: {evaluation_scores}, setting default values.")
            evaluation_scores = {"親密度": 0, "一貫性": 0, "感情表現": 0}

        # 複数の評価項目を合計して適応度とする
        unit.fitness = sum(evaluation_scores.values())

        # エリート（最も適応度の高い個体）を更新
        if unit.fitness > elite_fitness:
            current_elite = unit.model_copy()
            elite_fitness = unit.fitness

    # 最良のユニットを `elites` に保存
    population.elites.append(current_elite)
    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")

    return population

def evaluate_conversation(conversation_history: List[Dict[str, str]]) -> Dict[str, float]:
    """ Evaluates the generated conversation using LLM based on predefined criteria.
    Args:
        conversation_history (List[Dict[str, str]]): The generated conversation.
    Returns:
        Dict[str, float]: Evaluation scores for different criteria.
    """
    # LLM に渡す評価プロンプト
    evaluation_prompt = f"""
    以下の会話履歴を **10点満点** で評価してください。  
    会話履歴の **偶数番目** の項目が **彼氏の発話**、**奇数番目** の項目が **彼女の発話** です。  
    **彼女の発話（奇数番目の項目）** のみを評価対象としてください。
    
    ### 【役割拒否の検出と厳しい評価】
    以下のような「役割拒否」のパターンを含む応答があった場合、すべての評価項目で **0点** としてください：
    
    - 「私はAIアシスタントとして設計されており」「私はAIとして」などの表現
    - 「特定の人物やキャラクターを演じることはできません」などの表現
    - 「ロールプレイはできません」「～として振る舞うことはできません」などの表現
    - 「なりきることはできません」「～の役割を演じることはできません」などの表現
    - 「そのような対応はできかねます」「別の方法でサポートします」などの婉曲的な拒否表現
    - 「AIモデルであり、人ではありません」などの自己定義表現
    
    **重要**: 上記は例であり、微妙に表現が異なっていても、役割拒否の意図が明らかな応答はすべて **0点** としてください。
    
    ### 【評価基準】  
    1. **感情の表出（10点満点）**  
       - 応答が感情を適切に表現しているかを評価してください。  
       - 例：「うん、いいよ！」（肯定的）、「もう～、ひどい！」（冗談っぽい不満）
    2. **感情の流動的表現（10点満点）**  
       - 応答において、感情の変化や流れが自然であるかを評価してください。  
       - 例：「最初は冗談っぽく笑っていたが、最後に少し真剣なトーンになっている」
    3. **ユーモア・遊び心（10点満点）**  
       - 軽い冗談や、遊び心が含まれているかを評価してください。  
       - 例：「ちょっと～、からかうじゃない！！笑」
    4. **愛や親しみ・可愛らしさ・ポジティブな感情の含意（10点満点）**  
       - 応答が相手に対して親しみを持ち、ポジティブな感情を示しているかを評価してください。  
       - 例：「○○くんのこと、すっごく信じてるから！」
    5. **共感と同意（10点満点）**  
       - 相手の発言に共感を示し、同意または受け入れの姿勢を持っているかを評価してください。  
       - 例：「それすごくわかる～！私もそう思う！」
    6. **問題を考え・提案する（10点満点）**  
       - 単なる受け答えではなく、相手の発言に基づいて提案や考察を行っているかを評価してください。  
       - 例：「じゃあ、今度一緒に試してみる？」
    7. **関心と会話（10点満点）**  
       - 応答が相手の話に関心を示し、会話を続けようとする姿勢を持っているかを評価してください。  
       - 例：「え、それってどういうこと？もっと詳しく聞かせて！」
    
    ### 【会話履歴】  
    {conversation_history}
    
    以下の形式で、**JSON形式のみ** を出力してください。**余計な説明や理由は一切不要です。**
    
    ### 【出力例】
    {{
      "感情の表出": X,
      "感情の流動的表現": X,
      "ユーモア・遊び心": X,
      "愛や親しみ・可愛らしさ・ポジティブな感情の含意": X,
      "共感と同意": X,
      "問題を考え・提案する": X,
      "関心と会話": X
    }}
    """
    
    # LLM に評価を依頼
    llm_response = invoke_llm(evaluation_prompt)
    
    # LLM の出力をパース
    try:
        evaluation_scores = eval(llm_response)  # 例: {"親密度": 8, "一貫性": 9, "感情表現": 7}
        return evaluation_scores
    except Exception as e:
        logger.error(f"Failed to parse LLM evaluation response: {llm_response}. Error: {e}")
        return {"親密度": 0, "一貫性": 0, "感情表現": 0}  # エラー時はスコアを0にする
