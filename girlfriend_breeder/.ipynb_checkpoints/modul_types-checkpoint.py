import pydantic
from typing import List

class EvolutionUnit(pydantic.BaseModel):
    """ A single unit in the population, representing a system prompt variation.

    Attributes:
        'P': the task_prompt defining how the AI should behave as a girlfriend.
        'M': the mutation_prompt describing how to modify the task prompt.
        'fitness': the estimated effectiveness of the prompt.
        'history': the list of past versions of the task prompt.
    """
    P: str  # 彼女の振る舞いを定義するタスクプロンプト (Task Prompt)
    M: str  # タスクプロンプトをどう変化させるかを示すミューテーションプロンプト (Mutation Prompt)
    fitness: float  # どれだけ「彼女らしい」かのスコア (Fitness Score)
    history: List[str]  # 進化履歴（過去のプロンプトのリスト）

class Population(pydantic.BaseModel):
    """ The population model that evolves over generations.

    Attributes:
        'size': the number of individuals in the population.
        'age': the generation count.
        'problem_description': the target behavior we are optimizing.
        'elites': the best-performing individuals from past generations.
        'units': the current set of evolution units.
    """
    size: int  # 集団のサイズ
    age: int  # 世代数
    problem_description: str  # 最適化する対象（例：「彼女のように振る舞う AI のプロンプト」）
    elites: List[EvolutionUnit]  # エリート個体（過去の最高スコアのプロンプト）
    units: List[EvolutionUnit]  # 現在の集団の全個体
