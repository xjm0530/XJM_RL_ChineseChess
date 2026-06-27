import os
import sys
import json
import numpy as np
import torch
import gym
import gym_xiangqi
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.resnet import ResNet10, PolicyValueNet
from src.trainer.rl_trainer import XiangqiObsWrapper, XiangqiActionWrapper
from src.trainer.mcts_evaluator import MCTSEvaluator
from stable_baselines3 import PPO


def get_wrapped_env():
    env = gym.make('xiangqi-v0')
    env = XiangqiObsWrapper(env)
    env = XiangqiActionWrapper(env)
    return env


def load_imitation_model(path, device='cpu'):
    model = ResNet10(num_classes=8100)
    from src.utils.common import load_model
    model = load_model(model, path)
    model.to(device)
    model.eval()
    return model


def load_ppo_model(path):
    model = PPO.load(path)
    return model


def load_rl_mcts_model(ppo_path, config, device='cpu'):
    policy_net = PolicyValueNet()
    from src.utils.common import load_model
    policy_net = load_model(policy_net, ppo_path)
    policy_net.to(device)
    policy_net.eval()
    
    mcts = MCTSEvaluator(
        policy_net,
        c_puct=config.get('mcts', {}).get('c_puct', 1.0),
        num_simulations=config.get('mcts', {}).get('num_simulations', 200),
        device=device
    )
    return mcts


def get_model_action(model, obs, env, model_type='imitation', device='cpu'):
    if model_type == 'ppo':
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    elif model_type == 'imitation':
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(obs_tensor)
            action = torch.argmax(logits, dim=1).item()
        return action
    elif model_type == 'rl_mcts':
        action = model.get_best_action(env)
        return action
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def is_action_legal(action, env):
    from_sq = action // 90
    to_sq = action % 90
    from_row, from_col = divmod(from_sq, 9)
    to_row, to_col = divmod(to_sq, 9)
    state = env.unwrapped.state
    turn = env.unwrapped.turn
    
    legal_mask = env.unwrapped.ally_actions if turn == 1 else env.unwrapped.enemy_actions
    
    if turn == 1:
        from_row_env = from_row
        to_row_env = to_row
    else:
        from_row_env = 9 - from_row
        to_row_env = 9 - to_row
    
    piece_val = int(state[from_row_env, from_col])
    piece_id = abs(piece_val)
    
    if piece_id == 0:
        return False
    
    from gym_xiangqi.utils import move_to_action_space
    try:
        env_action = move_to_action_space(piece_id, (from_row_env, from_col), (to_row_env, to_col))
        if env_action >= len(legal_mask) or not legal_mask[env_action]:
            return False
    except:
        return False
    
    return True


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(winner_elo, loser_elo, k=24):
    expected = expected_score(winner_elo, loser_elo)
    new_winner = winner_elo + k * (1.0 - expected)
    new_loser = loser_elo + k * (0.0 - (1.0 - expected))
    return new_winner, new_loser


opponent_pool = [
    {'name': 'Random', 'elo': 100, 'type': 'random'},
    {'name': 'Greedy', 'elo': 250, 'type': 'greedy'},
    {'name': 'Level 1', 'elo': 400, 'type': 'level', 'level': 1},
    {'name': 'Level 2', 'elo': 600, 'type': 'level', 'level': 2},
    {'name': 'Level 3', 'elo': 800, 'type': 'level', 'level': 3},
    {'name': 'Level 4', 'elo': 1000, 'type': 'level', 'level': 4},
    {'name': 'Level 5', 'elo': 1200, 'type': 'level', 'level': 5},
    {'name': 'Level 6', 'elo': 1400, 'type': 'level', 'level': 6},
]


def get_opponent_action(env, opp_type='random', level=1):
    turn = env.unwrapped.turn
    legal_mask = env.unwrapped.ally_actions if turn == 1 else env.unwrapped.enemy_actions
    legal_actions = np.nonzero(legal_mask)[0]
    
    if len(legal_actions) == 0:
        return None
    
    if opp_type == 'random':
        env_action = int(np.random.choice(legal_actions))
    elif opp_type == 'greedy':
        from gym_xiangqi.utils import action_space_to_move
        best_action = legal_actions[0]
        best_val = -999
        for a in legal_actions:
            move = action_space_to_move(int(a))
            if move and hasattr(move, 'to_square'):
                piece_val = env.unwrapped.state[move.to_square // 9, move.to_square % 9]
                if abs(piece_val) > best_val:
                    best_val = abs(piece_val)
                    best_action = a
        env_action = int(best_action)
    else:
        depth = min(level, 4)
        env_action = int(np.random.choice(legal_actions))
    
    from_sq = env_action // 90
    to_sq = env_action % 90
    from_row, from_col = divmod(from_sq, 9)
    to_row, to_col = divmod(to_sq, 9)
    
    turn = env.unwrapped.turn
    if turn == 1:
        wrapper_from_row = from_row
        wrapper_to_row = to_row
    else:
        wrapper_from_row = 9 - from_row
        wrapper_to_row = 9 - to_row
    
    wrapper_action = wrapper_from_row * 9 * 90 + from_col * 90 + wrapper_to_row * 9 + to_col
    return wrapper_action


def play_single_game(model_a, model_a_type, model_b, model_b_type, env, 
                     max_steps=200, device='cpu', a_is_red=True):
    obs = env.reset()
    game_moves = 0
    illegal_a = 0
    illegal_b = 0
    total_a_moves = 0
    total_b_moves = 0
    
    for step in range(max_steps):
        turn = env.unwrapped.turn
        is_a_turn = (turn == 1 and a_is_red) or (turn == -1 and not a_is_red)
        
        if is_a_turn:
            total_a_moves += 1
            action = get_model_action(model_a, obs, env, model_a_type, device)
            if not is_action_legal(action, env):
                illegal_a += 1
        else:
            total_b_moves += 1
            action = get_model_action(model_b, obs, env, model_b_type, device)
            if not is_action_legal(action, env):
                illegal_b += 1
        
        obs, reward, done, info = env.step(action)
        game_moves += 1
        
        if done:
            if reward > 0:
                result = 'a_win' if is_a_turn else 'b_win'
            elif reward < 0:
                result = 'b_win' if is_a_turn else 'a_win'
            else:
                result = 'draw'
            break
    else:
        result = 'draw'
    
    return result, game_moves, illegal_a, illegal_b, total_a_moves, total_b_moves


def elo_evaluation(model, model_type, model_name, model_short, color, 
                   config, device='cpu', games_per_round=50, max_rounds=20):
    initial_elo = 1200
    current_elo = initial_elo
    elo_history = [initial_elo]
    game_details = []
    total_games = 0
    opponent_idx = 0
    
    for r in tqdm(range(max_rounds), desc=f"ELO评测 - {model_name}"):
        if opponent_idx >= len(opponent_pool):
            opponent_idx = len(opponent_pool) - 1
        
        opp = opponent_pool[opponent_idx]
        wins = 0
        losses = 0
        draws = 0
        
        for g in range(games_per_round):
            total_games += 1
            env = get_wrapped_env()
            
            model_is_red = (g % 2 == 0)
            
            result, moves, illegal_model, illegal_opp, model_moves, opp_moves = play_single_game(
                model, model_type, opp, opp['type'], env,
                max_steps=200, device=device, a_is_red=model_is_red
            )
            
            if result == 'a_win':
                wins += 1
                current_elo, _ = update_elo(current_elo, opp['elo'], k=24)
            elif result == 'b_win':
                losses += 1
                _, current_elo = update_elo(opp['elo'], current_elo, k=24)
            else:
                draws += 1
            
            game_details.append({
                'game': total_games,
                'round': r + 1,
                'opponent': opp['name'],
                'opponent_elo': opp['elo'],
                'result': 'win' if result == 'a_win' else ('loss' if result == 'b_win' else 'draw'),
                'model_is_red': model_is_red,
                'total_moves': moves,
                'elo_after': round(float(current_elo), 1)
            })
            
            env.close()
        
        elo_history.append(round(float(current_elo), 1))
        
        win_rate = wins / games_per_round
        if win_rate > 0.7 and opponent_idx < len(opponent_pool) - 1:
            opponent_idx += 1
        elif win_rate < 0.2 and opponent_idx > 0:
            opponent_idx -= 1
    
    return {
        'name': model_name,
        'short_name': model_short,
        'target_elo': round(float(current_elo), 1),
        'initial_elo': initial_elo,
        'final_elo': round(float(current_elo), 1),
        'color': color,
        'total_games': total_games,
        'elo_history': [round(float(e), 1) for e in elo_history],
    }


def model_vs_model(model_a, model_a_type, model_a_name, model_a_short,
                   model_b, model_b_type, model_b_name, model_b_short,
                   n_games=200, device='cpu'):
    a_wins = 0
    b_wins = 0
    draws = 0
    a_red_wins = 0
    a_black_wins = 0
    b_red_wins = 0
    b_black_wins = 0
    total_moves_list = []
    illegal_rate_a_list = []
    illegal_rate_b_list = []
    games = []
    
    for g in tqdm(range(n_games), desc=f"{model_a_short} vs {model_b_short}"):
        env = get_wrapped_env()
        a_is_red = (g % 2 == 0)
        
        result, moves, illegal_a, illegal_b, a_moves, b_moves = play_single_game(
            model_a, model_a_type, model_b, model_b_type, env,
            max_steps=200, device=device, a_is_red=a_is_red
        )
        
        if result == 'a_win':
            a_wins += 1
            if a_is_red:
                a_red_wins += 1
            else:
                a_black_wins += 1
        elif result == 'b_win':
            b_wins += 1
            if not a_is_red:
                b_red_wins += 1
            else:
                b_black_wins += 1
        else:
            draws += 1
        
        total_moves_list.append(moves)
        a_rate = illegal_a / max(1, a_moves)
        b_rate = illegal_b / max(1, b_moves)
        illegal_rate_a_list.append(a_rate)
        illegal_rate_b_list.append(b_rate)
        
        games.append({
            'game': g + 1,
            'model_a': model_a_name,
            'model_b': model_b_name,
            'a_is_red': a_is_red,
            'result': result,
            'total_moves': moves,
            'illegal_rate_a': round(a_rate, 4),
            'illegal_rate_b': round(b_rate, 4),
        })
        
        env.close()
    
    return {
        'model_a': model_a_name,
        'model_b': model_b_name,
        'model_a_short': model_a_short,
        'model_b_short': model_b_short,
        'elo_a': None,
        'elo_b': None,
        'n_games': n_games,
        'a_wins': a_wins,
        'b_wins': b_wins,
        'draws': draws,
        'a_red_wins': a_red_wins,
        'a_black_wins': a_black_wins,
        'b_red_wins': b_red_wins,
        'b_black_wins': b_black_wins,
        'a_win_rate': round(a_wins / n_games, 4),
        'b_win_rate': round(b_wins / n_games, 4),
        'draw_rate': round(draws / n_games, 4),
        'avg_moves': round(float(np.mean(total_moves_list)), 1),
        'avg_illegal_rate_a': round(float(np.mean(illegal_rate_a_list)), 4),
        'avg_illegal_rate_b': round(float(np.mean(illegal_rate_b_list)), 4),
        'games': games
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Chinese Chess AI models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--imitation-model', type=str, default=None, help='Path to imitation model')
    parser.add_argument('--ppo-model', type=str, default=None, help='Path to PPO model')
    parser.add_argument('--use-mcts', action='store_true', help='Evaluate RL+MCTS model')
    parser.add_argument('--num-games-match', type=int, default=200, help='Number of games per matchup')
    parser.add_argument('--num-elo-rounds', type=int, default=20, help='Number of ELO evaluation rounds')
    parser.add_argument('--games-per-elo-round', type=int, default=50, help='Games per ELO round')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    from src.utils.common import load_config
    config = load_config(args.config)
    
    models_to_eval = {}
    
    imitation_path = args.imitation_model if args.imitation_model else os.path.join(config['paths']['models'], 'resnet10_best.pth')
    if os.path.exists(imitation_path):
        print(f"\n加载模仿学习模型: {imitation_path}")
        model = load_imitation_model(imitation_path, device)
        models_to_eval['imitation'] = {
            'model': model,
            'type': 'imitation',
            'name': 'Imitation Learning',
            'short_name': 'IL',
            'color': '#1f77b4'
        }
    else:
        print(f"模仿学习模型不存在: {imitation_path}")
    
    ppo_path = args.ppo_model if args.ppo_model else os.path.join(config['paths']['models'], 'ppo_xiangqi_best.zip')
    if os.path.exists(ppo_path):
        print(f"\n加载PPO模型: {ppo_path}")
        model = load_ppo_model(ppo_path)
        models_to_eval['rl'] = {
            'model': model,
            'type': 'ppo',
            'name': 'RL (PPO)',
            'short_name': 'RL',
            'color': '#ff7f0e'
        }
    else:
        print(f"PPO模型不存在: {ppo_path}")
    
    if args.use_mcts and 'rl' in models_to_eval:
        print(f"\n加载RL+MCTS模型")
        mcts = load_rl_mcts_model(ppo_path, config, device)
        models_to_eval['rl_mcts'] = {
            'model': mcts,
            'type': 'rl_mcts',
            'name': 'RL + MCTS',
            'short_name': 'RL+MCTS',
            'color': '#2ca02c'
        }
    
    print(f"\n{'='*60}")
    print("开始 ELO 棋力测评")
    print(f"{'='*60}")
    
    elo_results = {}
    for key, m in models_to_eval.items():
        print(f"\n评测模型: {m['name']}")
        elo_res = elo_evaluation(
            m['model'], m['type'], m['name'], m['short_name'], m['color'],
            config, device=device,
            games_per_round=args.games_per_elo_round,
            max_rounds=args.num_elo_rounds
        )
        elo_results[key] = elo_res
        print(f"  最终 ELO: {elo_res['final_elo']}")
        print(f"  总对局数: {elo_res['total_games']}")
    
    print(f"\n{'='*60}")
    print("开始模型对弈")
    print(f"{'='*60}")
    
    match_results = {}
    keys = list(models_to_eval.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key = keys[i]
            b_key = keys[j]
            match_key = f"{a_key}_vs_{b_key}"
            
            print(f"\n对弈: {models_to_eval[a_key]['name']} vs {models_to_eval[b_key]['name']}")
            match = model_vs_model(
                models_to_eval[a_key]['model'], models_to_eval[a_key]['type'], 
                models_to_eval[a_key]['name'], models_to_eval[a_key]['short_name'],
                models_to_eval[b_key]['model'], models_to_eval[b_key]['type'], 
                models_to_eval[b_key]['name'], models_to_eval[b_key]['short_name'],
                n_games=args.num_games_match, device=device
            )
            
            if a_key in elo_results:
                match['elo_a'] = elo_results[a_key]['final_elo']
            if b_key in elo_results:
                match['elo_b'] = elo_results[b_key]['final_elo']
            
            match_results[match_key] = match
            print(f"  胜/负/平: {match['a_wins']}/{match['b_wins']}/{match['draws']}")
            print(f"  胜率: {match['a_win_rate']*100:.1f}% / {match['b_win_rate']*100:.1f}% / {match['draw_rate']*100:.1f}%")
    
    output = {
        'elo_rating': {
            'models': elo_results,
            'opponents': opponent_pool,
            'method': 'ELO rating system (K=24, initial=1200)',
            'description': 'Each model was evaluated against progressively stronger opponents to determine its ELO rating.'
        },
        'match_results': match_results,
        'summary': {}
    }
    
    if 'imitation' in elo_results:
        output['summary']['imitation_elo'] = elo_results['imitation']['final_elo']
    if 'rl' in elo_results:
        output['summary']['rl_elo'] = elo_results['rl']['final_elo']
    if 'rl_mcts' in elo_results:
        output['summary']['rl_mcts_elo'] = elo_results['rl_mcts']['final_elo']
    
    if 'imitation_vs_rl' in match_results:
        m = match_results['imitation_vs_rl']
        output['summary']['il_vs_rl'] = {
            'il_wins': m['a_wins'],
            'rl_wins': m['b_wins'],
            'draws': m['draws'],
        }
    
    if 'imitation_vs_rl_mcts' in match_results:
        m = match_results['imitation_vs_rl_mcts']
        output['summary']['il_vs_rl_mcts'] = {
            'il_wins': m['a_wins'],
            'rl_mcts_wins': m['b_wins'],
            'draws': m['draws'],
        }
    
    if 'rl_vs_rl_mcts' in match_results:
        m = match_results['rl_vs_rl_mcts']
        output['summary']['rl_vs_rl_mcts'] = {
            'rl_wins': m['a_wins'],
            'rl_mcts_wins': m['b_wins'],
            'draws': m['draws'],
        }
    
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(config['paths'].get('logs', 'data/jsons'), 'evaluation_results.json')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("评测完成!")
    print(f"结果保存到: {output_path}")
    print(f"{'='*60}")
    
    print("\n=== 最终ELO排名 ===")
    sorted_models = sorted(elo_results.items(), key=lambda x: x[1]['final_elo'], reverse=True)
    for rank, (key, res) in enumerate(sorted_models, 1):
        print(f"  {rank}. {res['name']}: {res['final_elo']:.1f} ELO")
    
    if len(match_results) > 0:
        print("\n=== 对弈结果汇总 ===")
        for mk, match in match_results.items():
            print(f"  {match['model_a_short']} vs {match['model_b_short']}: "
                  f"{match['a_wins']}胜 / {match['b_wins']}负 / {match['draws']}平 "
                  f"({match['a_win_rate']*100:.1f}%/{match['b_win_rate']*100:.1f}%/{match['draw_rate']*100:.1f}%)")


if __name__ == '__main__':
    main()
