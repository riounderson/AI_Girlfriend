import logging
import time
from typing import List

from rich import print
from conversation_orchestrator.orchestrator import orchestrator 
from utils.get_llm_result import get_llm_resut, invoke_llm  
from evaluation import evaluate_conversation  
from modul_types import EvolutionUnit, Population  
from mutation_operators import mutate 
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
                P=problem_description, 
                M=m,
                fitness=0,
                history=[problem_description]  
            ) for m in mutator_set
        ]
    }
    return Population(**data)


def init_run(population: Population, num_evals: int):
    """AWS Bedrock API を使い、`invoke_llm()` で逐次的にプロンプトを生成"""
    start_time = time.time()

    for i, unit in enumerate(population.units):    
        # プロンプトのテンプレートを作成
        template = f"""現在システムプロンプトに少し変化を加えるためのMutation Promptと言うものを用意しました。次のMutation Promptに従ってシステムプロンプトを修正してください。なお説明や理由などは一切不要です。新しいプロンプトのみを出力してください。
        Mutation Prompt: {unit.M} 
        システムプロンプト: {population.problem_description}
        """
        logger.info(f"Processing {i+1}/{len(population.units)}: {template}")

        # `invoke_llm()` を使ってプロンプトを生成
        prompt_result = invoke_llm(template)

        # エラー発生時は空文字をセット
        if prompt_result is None:
            prompt_result = ""

        unit.P = prompt_result

        time.sleep(1)

    end_time = time.time()
    logger.info(f"Prompt initialization done. {end_time - start_time}s")

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
        mutate(p)
        print("Mutation complete")
        evaluate_population(p, num_evals)  
        print("Evaluation complete")

    return p

