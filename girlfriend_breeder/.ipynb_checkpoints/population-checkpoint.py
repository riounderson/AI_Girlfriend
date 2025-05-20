import logging
import time
from typing import List

from rich import print
from conversation_orchestrator.orchestrator import orchestrator  # 彼氏・彼女の会話を生成
from utils.get_llm_result import get_llm_resut, invoke_llm  # AWS Bedrock で LLM を呼び出す
from evaluation import evaluate_conversation  # 評価関数
from modul_types import EvolutionUnit, Population  # データモデル
from mutation_operators import mutate  # 突然変異処理
from evaluation import evaluate_population

logger = logging.getLogger(__name__)

def create_population(mutator_set: List[str], problem_description: str) -> Population:
    """Creates an initial population of system prompts for Girlfriend Breeder.

    Args:
        mutator_set (List[str]): Mutation prompts to modify task prompts.
        problem_description (str): The initial system prompt defining how the AI should behave.

    Returns:
        Population: The initialized population.
    """
    data = {
        'size': len(mutator_set),
        'age': 0,
        'problem_description': problem_description,
        'elites': [],
        'units': [
            EvolutionUnit(
                P=problem_description,  # 動的に設定
                M=m,
                fitness=0,
                history=[problem_description]  # 初期プロンプトを履歴に保存
            ) for m in mutator_set
        ]
    }
    return Population(**data)


def init_run(population: Population, num_evals: int):
    """AWS Bedrock API を使い、`invoke_llm()` で逐次的にプロンプトを生成"""
    start_time = time.time()

    for i, unit in enumerate(population.units):    
        # プロンプトのテンプレートを作成
        template = f"{unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
        logger.info(f"Processing {i+1}/{len(population.units)}: {template}")

        # `invoke_llm()` を使ってプロンプトを生成
        prompt_result = invoke_llm(template)

        # エラー発生時は空文字をセット
        if prompt_result is None:
            prompt_result = ""

        # 取得したプロンプトを `EvolutionUnit` に保存
        unit.P = prompt_result

        # 1秒スリープしてAPI負荷を下げる
        time.sleep(1)

    end_time = time.time()
    logger.info(f"Prompt initialization done. {end_time - start_time}s")

    # 生成したプロンプトを評価
    evaluate_population(population, num_evals)
    
    return population

# def init_run(population: Population, model: Client, num_evals: int):
#     """ The first run of the population that consumes the prompt_description and 
#     creates the first prompt_tasks.
    
#     Args:
#         population (Population): A population created by `create_population`.
#     """

#     start_time = time.time()

#     prompts = []

#     for unit in population.units:    
#         template= f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
#         prompts.append(template)
    
 
#     results = model.batch_generate(prompts)

#     end_time = time.time()

#     logger.info(f"Prompt initialization done. {end_time - start_time}s")

#     assert len(results) == population.size, "size of google response to population is mismatched"
#     for i, item in enumerate(results):
#         population.units[i].P = item[0].text

#     _evaluate_fitness(population, model, num_evals)
    
#     return population

def run_for_n(n: int, population: Population, num_evals: int):
    """ Runs the genetic algorithm for n generations.

    Args:
        n (int): Number of generations.
        population (Population): The evolving population.
        num_evals (int): Number of evaluations per generation.

    Returns:
        Population: The evolved population.
    """
    p = population
    for i in range(n):
        print(f"================== Generation {i} ==================")
        mutate(p)  # 突然変異（プロンプトを改良）
        print("Mutation complete")
        evaluate_population(p, num_evals)  # 適応度評価を `evaluation.py` に移動
        print("Evaluation complete")

    return p

