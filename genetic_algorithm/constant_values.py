import yaml

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# グローバル変数として設定を読み込む
config = load_config('config.yaml')

# 設定値の格納
POPULATION_SIZE = config["population_size"]
NUM_GENERATIONS = config["num_generations"]
MUTATION_RATE = config["mutation_rate"]
NUM_GENES = config["num_genes"]


