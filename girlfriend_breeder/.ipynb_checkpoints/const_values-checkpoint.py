import yaml
import os

def load_config():
    # `const_values.py` のあるディレクトリを基準に `setup_info.yml` を探す
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(base_dir)
    file_path = os.path.join(base_dir, "setup_info.yml") 

    # `setup_info.yml` のパスを確認
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"setup_info.yml が見つかりません: {file_path}")

    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

setup_info = load_config()

API_URL = setup_info['API_URL']

if __name__ == "__main__":
    print(setup_info)