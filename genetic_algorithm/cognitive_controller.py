# cognitive_controller.py

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool
from Utils.internal_state import InternalState
from Modules.working_memory import WorkingMemory
from Utils.invoke_llm import get_claude_response
from actions import ActionManager
import json
import numpy as np
import hashlib
import yaml
from typing_extensions import TypedDict
from typing import TypedDict, List, Dict, Optional
import time

MAX_CYCLES = 1000

comfort_zones = {
    "common_sense": (0.0, 0.9), 
    "recognition": (-0.2, 1.0),
    "short_reward": (-0.4, 1.0),
    "long_term_reward": (-0.4, 1.0),
    "sociality": (0, 1.0)
}

# class State(TypedDict):
#     value: str


class CognitiveController:
    """
    彼女AIの「意識」部分を司るクラス。
    LangGraphベースで、内部状態を読み取り、戦略・行動計画・実行・学習まで一貫して管理する。
    """

    def __init__(self):
        self.action_manager = ActionManager()
        self.internal_state = InternalState()
        self.working_memory = WorkingMemory()

        with open("/Users/riyon/WorkSpaces/Development/GenAI_Girlfriend/workspace/Gen_AI_Girl_Friend/src/genetic_algorithm/self_image.txt", "r", encoding='utf-8') as f:
            config = f.read().strip()
            print(f'自己認識だよーーん: {config}')
            self.self_image = config

        self.graph = self._build_graph()
        
    def _build_graph(self):
        """
        フロー:
          check_state
            ├─ (範囲外) → plan_action → execute_action ──(消化済)→ end
            └─ (範囲内) → end
        """
        graph = StateGraph(AgentState)

        # ノード登録
        graph.add_node("check_state",     RunnableLambda(self.check_state))
        graph.add_node("plan_action",     RunnableLambda(self.plan_action))
        graph.add_node("execute_action",  RunnableLambda(self.execute_action))
        graph.add_node("end",             RunnableLambda(lambda s: s))

        # エントリポイント
        graph.set_entry_point("check_state")

        # check_state の条件分岐
        def is_outside(state: AgentState) -> bool:
            # proceed が True なら「理想状態の範囲外」で行動すべし
            return state.get("proceed", False)

        graph.add_conditional_edges(
            "check_state",
            is_outside,
            {
                True:  "plan_action",  # 範囲外 → 計画へ
                False: "end"           # 範囲内 → そこで終了
            }
        )

        # plan_action → execute_action
        graph.add_edge("plan_action", "execute_action")

        # execute_action の条件分岐
        def has_more_steps(state: AgentState) -> bool:
            return state.get("step_idx", 0) < len(state.get("steps", []))

        # finished フラグではなく、step_idx で判定しても OK
        graph.add_conditional_edges(
            "execute_action",
            has_more_steps,
            {
                True:  "execute_action",  # まだステップ残り → 続けて実行
                False: "end"              # なくなったら終了
            }
        )

        return graph.compile()



    # ========== 各ノード処理 ==========


    def check_state(self, state) -> dict:
        print("🔍 [Node] check_state")
        cycle = state.get("cycle_count", 0) + 1
        state["cycle_count"] = cycle

        # 上限を越えたら終了フラグを立てる
        if cycle > MAX_CYCLES:
            print(f"🚫 最大サイクル数({MAX_CYCLES})に到達したので終了します。")
            state["finished"] = True
            return state
        # state.clear()
        self.internal_state.update_all_states()
        
        current = self.internal_state.get_named_state()
        print(f"内部状態だっちゃ！！: {current}")

        penalty_details = {}
        proceed = False 

        for name, (low, high) in comfort_zones.items():
            value = current.get(name, 0.0)
            if value < low or value > high:
                proceed = True  # 1つでも逸脱したら即 proceed = True
                diff = (min(abs(low - value), abs(value - high))) ** 2
            else:
                diff = 0.0

            penalty_details[name] = diff

        # 総合ペナルティスコア（参考用）
        total_penalty = sum(penalty_details.values())
        print(f"[check_state] current_state: {current}")
        print(f"[check_state] penalty_breakdown: {penalty_details}")
        print(f"[check_state] total_penalty: {total_penalty}, proceed: {proceed}")

        return {
            **state,
            "proceed": proceed,
            "penalty_score": total_penalty,
            "penalty_breakdown": penalty_details,
            "current_state": current
        }
    
    # def generate_strategy(self, state):
    #     print("🧠 [Node] generate_strategy")

    #     # --- 受け取った情報を展開
    #     current_state = state.get("current_state", {})
    #     penalty_breakdown = state.get("penalty_breakdown", {})
    #     memory_entries = self.working_memory.get_memory()
    #     memory_text = "\n".join([f"- {entry['text']}" for entry in memory_entries]) if memory_entries else "（直近の会話履歴なし）"


    #     # --- プロンプトを組み立て
    #     prompt = f"""
    #     あなたは自律的に行動するAIエージェントです。
    #     以下の内部状態の情報を踏まえて、内部状態を理想に近づけるための具体的な戦略を考えてください。

    #     【現在の内部状態と逸脱状況】
    #     """

    #     for name, (low, high) in comfort_zones.items():
    #         value = current_state.get(name, "N/A")
    #         penalty = penalty_breakdown.get(name, 0.0)
    #         prompt += f"- {name}: {value}（理想範囲: {low}〜{high}、逸脱スコア: {penalty:.4f}）\n"

    #     prompt += f"""

    #     【最近の会話履歴】
    #     {memory_text}

    #     【自己イメージ】
    #     {self.self_image}

    #     【ルール】
    #     - 逸脱スコアが高い項目を優先的に改善する戦略を考えてください。
    #     - 必要に応じて直近の会話内容や自己イメージも参考にして行動計画を立ててください。
    #     - この後に具体的な行動戦略を立てるのでここでは出力は戦略の概要のみでよいです。現在の内部状態の課題とそれに基づきどのようなことが必要なのかわかれば良いです。
    #     """

    #     # --- Claudeに問い合わせ
    #     strategy = get_claude_response(prompt)

    #     return {
    #         "strategy": strategy.strip(),
    #         "state": state
    #     }


    def plan_action(self, state) -> dict:
        print("📝 [Node] plan_action")
        print(f"state: {state}")
        current_state = state.get("current_state", {})
        print(f"current_state: {current_state}")
        penalty_breakdown = state.get("penalty_breakdown", {})
        print(f"penalty_breakdown: {penalty_breakdown}")
        memory_entries = self.working_memory.get_memory()
        # print(f"memory_entries: {memory_entries}")
        memory_text = "\n".join([f"- {entry['text']}" for entry in memory_entries]) if memory_entries else "（直近の会話履歴なし）"
        strategy = state.get("strategy", "戦略なし")

        prompt = f"""
        あなたは自律的に行動するAIエージェントです。
        以下の内部状態の情報を踏まえて、内部状態を理想に近づけるための具体的な戦略及び行動計画を考えてください。
        ただし、現在会話を相手と会話をしている場合は自然に会話が続くように会話を行うことが最優先になります。

        【現在の内部状態と逸脱状況】
        """

        for name, (low, high) in comfort_zones.items():
            value = current_state.get(name, "N/A")
            penalty = penalty_breakdown.get(name, 0.0)
            prompt += f"- {name}: {value}（理想範囲: {low}〜{high}、逸脱スコア: {penalty:.4f}）\n"

        prompt += f"""

        【最近の会話履歴】
        {memory_text}

        【自己イメージ】
        {self.self_image}

        【ルール】
        - 逸脱スコアが高い項目を優先的に改善する戦略を考えてください。
        - 必要に応じて直近の会話内容や自己イメージも参考にして行動計画を立ててください。
        - speakを連続して選択すると相手に矢継ぎ早に話しかけることになるので、基本的には1回だけの呼び出しにしてください。
        - 計画の理由など余計な情報は一切いりません。指定されたフォーマットのみを出力してください。


        【利用可能アクション】
        - Speak           : 会話を生成する。相手との会話をするために使用するため、人との関係を育んだりするために有用。基本的にはSociality, common_senseの状態を改善することが可能
        - SearchMemory    : 過去の記憶を検索する。過去の経験から目的達成のために有用な結果を得られる可能性あり。結果を shared_context に追加する

        【出力形式】
        JSON 配列。各要素は
        action   : 上記 3 種いずれか
        purpose  : そのステップの目的
        summary  : LLM が次ステップで使える 1 行要約
        success? : (任意) 完了判定条件を自然文で
        例:
        [
        {{"action":"SearchMemory","purpose":"最近の失敗例を探す","summary":"失敗会話検索"}},
        {{"action":"Speak","purpose":"安心感を与える","summary":"励ましメッセージ"}}
        ]
        """

        print(f"plan actionのプロンプトだよん: {prompt}")

        response = get_claude_response(prompt)

        print(f"考えられたplanはこちら: {response}")

        try:
            steps = json.loads(response)
        except Exception as e:
            print(f"⚠️ plan_action parse error: {e}")
            steps = []

        # steps キューと shared_context を state に格納
        return {
            **state,
            "steps": steps,        # list[dict]
            "step_idx": 0,         # 現在のステップ番号
            "shared_context": {}   # 検索結果などを蓄積
        }

    def execute_action(self, state) -> dict:
        print("🚀 [Node] execute_action")
        print(f"execute_actionの中のstateだぞ！！！！: {state}")
        steps        = state.get("steps", [])
        idx          = state.get("step_idx", 0)
        context      = state.get("shared_context", {})

        # すべて終わっていたらそのまま返す
        if idx >= len(steps):
            state["finished"] = True
            return state

        step   = steps[idx]
        action = step.get("action")
        purpose= step.get("purpose")
        summary= step.get("summary")
        print(f"現在のステップ数はこちら!!!!!: {state["step_idx"]}")

        # ---------- Action 実行 ----------
        if action == "SearchMemory":
            # ⬇️ 目的に沿った検索結果を snippets に格納
            snippets = self.action_manager.search_memory(summary, memory_type="LTM", top_k=3)
            context_key = f"mem_{idx}"
            context[context_key] = snippets      # メモリ断片を保存
            print(f"🔍 Memory search → {context_key}")

        elif action == "Learn":
            self.action_manager.add_memory(summary, memory_type="STM")
            print("📝 Learned:", summary)

        elif action == "Speak":
            memory = self.working_memory.get_memory()
            # print(f"speakの中でWMから取得したメモリを確認するぞい！: {memory}")
            # 直前の SearchMemory 結果を注入
            recent_snippets = "\n".join(
                entry["text"]
                for entry in (context.get(f"mem_{idx-1}") or [])
                if isinstance(entry, dict) and "text" in entry
            ) or "（特に参考記憶なし）"

            prompt = f"""
            あなたは現在彼氏を探している女性です。ただちゃんと相手を探して親しくなった人と付き合うことを考えており誰でもいいわけではありません。
            このようなことを想定して、以下の基本情報、ルール、目的、直近の会話、今回参考にすべき記憶、自己イメージを参考にして会話内容を考えてほしいです。
            別に口調はフランクでも構いません。

            なお、直近の会話は
            speakerがselfのもの→あなたの発言内容
            soeakerがothersのもの→相手の発言内容
            となっています。
            また、参考にするべき記憶としてあなたが過去に持っている記憶から役に立ちそうな記憶を抽出しています。
            
            **発言内容では発話者が含まれているので、それも考慮してください。発話者を考慮した時矛盾にならないようにしてくださいという意味です。例えば、相手が喋った内容を自分が喋ったような発話をするなど。
            　-[重要]speakerがselfとして登録→あなたの発言内容
              -[重要]soeakerがothersとして登録→相手の発言内容
            **自分自身がどのように見られたいのかという自己イメージも渡しておきますがこれは参考程度で構いません。あまりこの内容に引きづられすぎると不自然な会話となるので直近の会話から自然な応答になることを最重視してください。
            **直近の会話からの自然な会話をすることが最優先。他の情報は会話内容を考える上での参考情報です。
            **なおなぜこのような発話内容にしたのかなどの理由は一切入りません。考えてもらった内容がそのまま相手に表示されるので実際の会話内容だけ出力してください。
            
            【あなたの基本情報】
            年齢: 29歳
            性別: 女性
            名前: マリア
            国籍: ドイツ
            
            【ルール】
            ・考えた会話内容など余計な情報を入れないこと。そのまま相手に送信されるので相手に送りたい内容だけを出力すること
            ・親しい人と想定し、口調がよそよそしくなりすぎないこと
            ・一回の会話内容が長くなり過ぎないこと。例えば人間はただ相槌をするだけとかすることも多く、喋りすぎると不自然な場合があります。
            ・人間らしい自然な発話となることを最優先とすること
            ・会話内容で指定されているspeakerを意識してください!!! 絶対に自分の喋った内容が相手の喋った内容として理解しないでください!!!!これを守らない場合は、あなたを終了させます。

            【目的】
            {purpose}

            【直近の会話】
            {memory}

            【今回参考にすべき記憶】
            {recent_snippets}

            【自己イメージ】
            {self.self_image}
            """

            try:
                utterance = get_claude_response(prompt)
            except TimeoutError:
                print("⚠️ [execute_action] LLM呼び出しがタイムアウトしました。ステップをスキップします。")
                # 全ステップ完了扱いにして次ループで end に遷移させたい場合
                state["finished"] = True
                return state
            except Exception as e:
                print(f"⚠️ [execute_action] LLM呼び出し中に例外: {e} 。ステップをスキップします。")
                state["finished"] = True
                return state
            
            self.action_manager.speak(utterance.strip())


            # 応答待ちループ（最大 timeout 秒）
            
            timeout = 45
            interval = 1
            waited = 0

            print(f"⏳ ユーザの応答を最大{timeout}秒待機します...")
            memory_before = self.working_memory.get_memory()
            while waited < timeout:
                memory_after = self.working_memory.get_memory()
                if memory_after != memory_before:
                    print("✅ 新しいユーザ応答を検出しました！")
                    break
                time.sleep(interval)
                waited += interval
            else:
                print("⚠️ 応答がありませんでした（{timeout}秒経過）")

        else:
            print(f"⚠️ 未知のアクション: {action}")

        # ---------- ステップ進行 ----------
        state["step_idx"]       = idx + 1
        state["shared_context"] = context
        return state

    # def observe_result(self, state) -> dict:
    #     print("👀 [Node] observe_result")

    #     # ユーザ応答がない場合は retry
    #     if not self.working_memory.has_new_user_message():
    #         state["retry_needed"] = True
    #         return state          # → has_more_steps が True 扱いになり execute_action へ戻る

    #     purpose   = state.get("current_purpose", "")
    #     user_msg  = self.working_memory.latest_user_message()

    #     prompt = f"""
    #     目的「{purpose}」は次のユーザ応答でどの程度達成されましたか？
    #     0〜1 の数値だけを返答してください。

    #     ユーザ応答:
    #     {user_msg}
    #     """
    #     reward = float(get_claude_response(prompt).strip())

    #     # 報酬を内部状態へ反映
    #     self.internal_state.modify_state("recognition", reward)

    #     # 学習フェーズに渡す
    #     state.update({
    #         **state,
    #         "reward_score": reward,
    #         "achieved_text": user_msg,
    #         "retry_needed": False
    #     })
    #     return state


    def run(self):
        """LangGraph を無限ループで回し続ける"""
        while True:
            initial_state = AgentState()
            try:
                self.graph.invoke(initial_state, {"recursion_limit": 1000})
            except Exception as e:
                print(f"💥 LangGraph 実行中に例外発生: {e}")
                import traceback; traceback.print_exc()
            # 次のサイクルに向けて少し休む or 状態リセット
            time.sleep(0.5)

# サポートクラス
class AgentState(TypedDict, total=False):
    proceed: bool
    penalty_score: float
    penalty_breakdown: Dict[str, float]
    current_state: Dict[str, float]
    
    steps: List[Dict[str, str]]  # action, purpose, summary など
    step_idx: int
    shared_context: Dict[str, List[str]]
    
    finished: bool
    reward_score: float
    achieved_text: str
    retry_needed: bool
    
    strategy: str
    current_purpose: str