import random
import json
import logging
from typing import List
from modul_types import Population, EvolutionUnit
from utils.get_llm_result import get_llm_resut, invoke_llm
from utils.response_checker import resulf_judge

logger = logging.getLogger(__name__)

# **理想の会話例データの読み込み**
def load_ideal_conversation_examples(filepath: str = "../data/ideal_conversation_example.json") -> List[dict]:
    """ Reads ideal conversation examples from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        List[dict]: List of ideal conversation examples.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load ideal conversation examples: {e}")
        return []

ideal_conversations = load_ideal_conversation_examples()

# **0次超変異（Mutation Prompt のみを変異）**
def zero_order_hypermutation(unit: EvolutionUnit, **kwargs) -> EvolutionUnit:
    """ Mutates the existing mutation prompt (M) using LLM.

    Returns:
        EvolutionUnit: The evolution unit with a new mutation prompt.
    """
    mutation_prompt = f"""
    現在システムプロンプトに少し変化を加えるためのMutation Promptと言うものを作成しています。
    こちらにMutation Promptの例を提示するのでこれを参考にして改善してください: {unit.M}
    なお説明や理由などは一切不要です。新しいプロンプトのみを出力してください。
    """
    unit.M = invoke_llm(mutation_prompt)
    return unit

# **理想の会話例を元にプロンプト (`P`) を改良**
def working_out_task_prompt(unit: EvolutionUnit, **kwargs) -> EvolutionUnit:
    """ Uses an ideal conversation example to refine the task prompt.

    Returns:
        EvolutionUnit: The evolution unit with an improved prompt.
    """
    if not ideal_conversations:
        logger.warning("No ideal conversation examples available. Skipping mutation.")
        return unit

    example = random.choice(ideal_conversations)

    refinement_prompt = f"""
    以下の会話例を参考にして、このような会話を行うためのシステムプロンプトを生成してください。

    [理想の会話例]
    {example['conversation']}

    なお説明や理由などは一切不要です。新しいプロンプトのみを出力してください。
    """
    print(f"変異元プロンプト: {unit.P}")

    try_num = 0
    max_retry = 5
    
    while try_num <= max_retry:
        unit.P = invoke_llm(refinement_prompt)
        result = resulf_judge(unit.P)
        if result and result == "True":
            break
        else:
            try_num += 1

    print(f'変異後プロンプト: {unit.P}')

    return unit

# **1次変異（Mutation Prompt を適用）**
def first_order_prompt_gen(unit: EvolutionUnit, **kwargs) -> EvolutionUnit:
    """Concatenate the mutation prompt M to the parent task-prompt P and pass it to the LLM to produce P'

    Returns:
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    try_num = 0
    max_retry = 5
    print(f"変異元プロンプト: {unit.P}")
    
    while try_num <= max_retry:
        unit.P = invoke_llm(unit.M + " " + unit.P + "なお説明や理由などは一切不要です。新しいプロンプトのみを出力してください。")
        result = resulf_judge(unit.P)
        if result and result == "True":
            break
        else:
            try_num += 1

    print(f'変異後プロンプト: {unit.P}')

    return unit

# **系統変異（エリートの履歴を活用）**
def lineage_based_mutation(unit: EvolutionUnit, elites: List[EvolutionUnit], **kwargs) -> EvolutionUnit:
    """Using the stored history of best units, provide the LLM this list in chronological order to produce a novel prompt as continuation.

    Returns:
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    HEADING = """以下は品質の良い順に並んだシステムプロンプトです。これを参考にして新たなプロンプトを作成してください。 \n 
    なお説明や理由などは一切不要です。新しいプロンプトのみを出力してください。"""
    ITEMS = "\n".join(["{}. {}".format(i+1, x.P) for i, x in enumerate(elites)])


    try_num = 0
    max_retry = 5
    print(f"変異元プロンプト: {unit.P}")
    while try_num <= max_retry:
        unit.P = invoke_llm(HEADING + ITEMS)
        result = resulf_judge(unit.P)
        if result and result == "True":
            break
        else:
            try_num += 1

    print(f'変異後プロンプト: {unit.P}')
    
    return unit

def prompt_crossover(unit1: EvolutionUnit, unit2: EvolutionUnit) -> EvolutionUnit:
    """Combines parts of two prompts to create a new prompt.

    Args:
        unit1 (EvolutionUnit): First unit (parent 1).
        unit2 (EvolutionUnit): Second unit (parent 2).

    Returns:
        EvolutionUnit: The new unit with a crossover prompt.
    """
    sentences1 = unit1.P.split("。")  # 文を分割
    sentences2 = unit2.P.split("。")

    print(f"クロスオーバー1: {sentences1}")
    print(f"クロスオーバー2: {sentences2}")

    # 各プロンプトからランダムに半分ずつ選択
    half1 = random.sample(sentences1, len(sentences1) // 2)
    half2 = random.sample(sentences2, len(sentences2) // 2)

    # 新しいプロンプトを作成
    new_prompt = "。".join(half1 + half2) + "。"

    # 新しい EvolutionUnit を作成
    return EvolutionUnit(P=new_prompt, M=random.choice([unit1.M, unit2.M]), fitness=0, history=[new_prompt])


def context_shuffling(unit: EvolutionUnit) -> EvolutionUnit:
    """Randomly shuffles sentences or phrases within a prompt to introduce variation.

    Args:
        unit (EvolutionUnit): The unit to be shuffled.

    Returns:
        EvolutionUnit: The shuffled unit.
    """
    sentences = unit.P.split("。")
    random.shuffle(sentences)  # 文をランダムに並び替え
    shuffled_prompt = "。".join(sentences) + "。"

    unit.P = shuffled_prompt
    print(f"シャッフル後のプロンプト: {unit.P}")
    return unit

# **変異オペレーターのリスト**
MUTATORS = [
    first_order_prompt_gen,
    lineage_based_mutation,
    zero_order_hypermutation,
    working_out_task_prompt
]

POST_MUTATORS = [
    prompt_crossover,
    context_shuffling
]

# **メインの変異関数**
def mutate(population: Population) -> Population:
    """Select and apply a random mutator to the losing unit in each pair."""

    indices = [i for i in range(len(population.units))]
    random.shuffle(indices)
    pairs = [indices[2*x:2*x+2] for x in range(len(indices) // 2)]

    for i in range(len(pairs)):
        first_unit = population.units[pairs[i][0]]
        second_unit = population.units[pairs[i][1]]

        FIRST_WON = first_unit.fitness >= second_unit.fitness
        mutation_input = second_unit if FIRST_WON else first_unit

        data = {
            'unit': mutation_input,
            'elites': population.elites
        }

        random_mutator = random.sample(MUTATORS, 1)[0]
        print(f"MUTATING: {mutation_input} with {random_mutator.__name__}")

        random_mutator(**data)

    return population
