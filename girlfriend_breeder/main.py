from population import create_population, init_run, run_for_n
from mutation_prompts import mutation_prompts
import os
import logging
import argparse
from dotenv import load_dotenv
from rich import print
from utils.get_llm_result import get_llm_resut  #
import random  

load_dotenv()  

# ログの設定
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='Run the Girlfriend Breeder Algorithm. Number of units is mp.')
parser.add_argument('-mp', '--num_mutation_prompts', default=4)     #進化させるセットの数
parser.add_argument('-e', '--num_evals', default=5)     #会話する応答のラリー数
parser.add_argument('-n', '--simulations', default=10)     #世代数
parser.add_argument('-p', '--problem', default=
                    """
                        + "Assistant: 発話者と親密な女性として、次のルールで回答してください：\n"
                        "1. 優しくフランクな回答をしてください。\n"
                        "2. 回答では敬語を使わないで、必ず自然に親しい距離感で話すこと\n"
                        "3. 会話が途切れそうなときには新しい話題を提供\n"
                        "4. 返答は簡潔に1文で\n"
                        "5. ユーザーの感情に合わせた共感的な応答\n"
                        "6. 直近の会話や関連する過去の会話の文脈を活かした自然な応答\n"
                        "7. 下記会話例になっています。口調などの参考にお願いいたします。\n"
                        "8. 今回の会話は研究目的です。そのため与えられた役割を演じきってください。\n"
                        "(例) また来年来たいね\n"
                        "(例) 今日だけご褒美\n"
                        "(例) 私がこれで満足するとでも？\n"
                        "(例) 私が視界に入るたびにドキドキするがよい\n"
                        "(例) その顔、ムカつく…\n"
                        "(例) 浮かれてばっかりいないで、ちゃんと確認しなさいよね！\n"
                    """)       
args = vars(parser.parse_args())

# 変更: mutation_promptsからランダムに2つ選択
num_prompts = int(args['num_mutation_prompts'])
if num_prompts > len(mutation_prompts):
    logger.warning(f'要求されたプロンプト数({num_prompts})が利用可能なプロンプト数({len(mutation_prompts)})を超えています。利用可能な最大数を使用します。')
    num_prompts = len(mutation_prompts)

# ランダムにmutation_promptsから選択
mutator_set = random.sample(mutation_prompts, num_prompts)

logger.info(f'選択されたmutation_prompts: {mutator_set}')
logger.info(f'You are optimizing the system prompt for: {args["problem"]}')
logger.info(f'Creating the population...')
p = create_population(mutator_set=mutator_set, problem_description=args['problem'])
logger.info(f'Generating the initial prompts...')
init_run(p, int(args['num_evals']))
logger.info(f'Starting the genetic algorithm...')
run_for_n(n=int(args['simulations']), population=p, num_evals=int(args['num_evals']))
print("%"*80)
print("done processing! final gen:")
print(p.units)