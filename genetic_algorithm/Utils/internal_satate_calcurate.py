import numpy as np

def calculate_state_difference(current_state, ideal_state):
    """
    現在の内部状態と理想の内部状態の数値的な差分を計算する。
    :param current_state: 現在の内部状態（VAE から出力された数値ベクトル）
    :param ideal_state: 理想的な内部状態（基準となる数値ベクトル）
    :return: 差分ベクトル, 差の合計スコア, ズレが大きい要素のインデックス
    """
    # 差分の計算
    difference = np.array(current_state) - np.array(ideal_state)
    
    # ユークリッド距離を計算（全体のズレをスカラー値で表現）
    total_difference = np.linalg.norm(difference)
    
    # 各要素の絶対値の大きい順に並べる（最もズレが大きいものを特定）
    sorted_indices = np.argsort(np.abs(difference))[::-1]  # 降順
    
    return difference, total_difference, sorted_indices

def interpret_difference(difference, feature_names):
    """
    差分ベクトルを解釈し、どの要素が最もズレているかを特定する。
    :param difference: 差分ベクトル
    :param feature_names: 各潜在変数の名前（例: ["mood", "energy", "conversation", "interest", "stress"]）
    :return: 解釈された行動指針（ズレの大きい特徴リスト）
    """
    actions = []
    
    # 各特徴のズレを見て、特にズレが大きい場合に行動の指針を作る
    for i, value in enumerate(difference):
        if value < -0.5:
            actions.append(f"{feature_names[i]} が低すぎる")
        elif value > 0.5:
            actions.append(f"{feature_names[i]} が高すぎる")
    
    return actions
