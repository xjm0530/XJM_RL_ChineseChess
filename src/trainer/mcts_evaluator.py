import numpy as np
import torch
import gym
import gym_xiangqi
from gym_xiangqi.utils import move_to_action_space, action_space_to_move
from src.utils.common import encode_board_from_state, flip_perspective, get_device


class MCTSNode:
    def __init__(self, state, turn, parent=None):
        self.state = state
        self.turn = turn
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def select(self, c_puct=1.0):
        best_score = -np.inf
        best_action = None
        
        for action, child in self.children.items():
            u = c_puct * child.prior * np.sqrt(self.visits) / (1 + child.visits)
            score = child.value() + u
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def expand(self, policy, legal_actions_8100):
        for action in legal_actions_8100:
            if action not in self.children:
                self.children[action] = MCTSNode(None, None, parent=self)
                self.children[action].prior = policy[action] if action < len(policy) else 0.0
    
    def update(self, value):
        self.visits += 1
        self.value_sum += value


class MCTS:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()
        self.c_puct = config.get('mcts', {}).get('c_puct', 1.0)
    
    def _get_legal_actions_8100(self, env):
        state = env.unwrapped.state
        turn = env.unwrapped.turn
        
        legal_mask = env.unwrapped.ally_actions if turn == 1 else env.unwrapped.enemy_actions
        env_actions = np.where(legal_mask > 0)[0]
        
        actions_8100 = []
        for env_act in env_actions:
            move_info = action_space_to_move(env_act)
            piece_id = move_info[0]
            from_pos = move_info[1]
            to_pos = move_info[2]
            
            if turn == 1:
                from_row, from_col = from_pos[0], from_pos[1]
                to_row, to_col = to_pos[0], to_pos[1]
            else:
                from_row = 9 - from_pos[0]
                from_col = from_pos[1]
                to_row = 9 - to_pos[0]
                to_col = to_pos[1]
            
            from_sq = from_row * 9 + from_col
            to_sq = to_row * 9 + to_col
            action_8100 = from_sq * 90 + to_sq
            actions_8100.append((action_8100, env_act))
        
        return actions_8100
    
    def _convert_action_8100_to_env(self, action_8100, env):
        from_sq = action_8100 // 90
        to_sq = action_8100 % 90
        from_row, from_col = divmod(from_sq, 9)
        to_row, to_col = divmod(to_sq, 9)
        
        state = env.unwrapped.state
        turn = env.unwrapped.turn
        
        if turn == 1:
            from_row_env = from_row
            to_row_env = to_row
            legal_mask = env.unwrapped.ally_actions
        else:
            from_row_env = 9 - from_row
            to_row_env = 9 - to_row
            legal_mask = env.unwrapped.enemy_actions
        
        piece_val = int(state[from_row_env, from_col])
        piece_id = abs(piece_val)
        
        if piece_id == 0:
            legal_env_actions = np.nonzero(legal_mask)[0]
            if len(legal_env_actions) > 0:
                return int(legal_env_actions[0])
            return 0
        
        env_action = move_to_action_space(piece_id, (from_row_env, from_col), (to_row_env, to_col))
        if 0 <= env_action < len(legal_mask) and legal_mask[env_action]:
            return int(env_action)
        
        legal_env_actions = np.nonzero(legal_mask)[0]
        if len(legal_env_actions) > 0:
            return int(legal_env_actions[0])
        return 0
    
    def _encode_state(self, state, turn):
        encoded = encode_board_from_state(state)
        if turn != 1:
            encoded = flip_perspective(encoded)
        return encoded
    
    def _is_game_over(self, env):
        turn = env.unwrapped.turn
        legal_mask = env.unwrapped.ally_actions if turn == 1 else env.unwrapped.enemy_actions
        if np.sum(legal_mask) == 0:
            return True
        
        state = env.unwrapped.state
        red_king = np.any(state == 1)
        black_king = np.any(state == -1)
        if not red_king or not black_king:
            return True
        
        return False
    
    def _get_winner(self, env):
        state = env.unwrapped.state
        red_king = np.any(state == 1)
        black_king = np.any(state == -1)
        
        if not red_king and not black_king:
            return 0
        if not red_king:
            return -1
        if not black_king:
            return 1
        
        turn = env.unwrapped.turn
        legal_mask = env.unwrapped.ally_actions if turn == 1 else env.unwrapped.enemy_actions
        if np.sum(legal_mask) == 0:
            return -turn
        
        return 0
    
    def run(self, env, num_simulations=200):
        root_state = env.unwrapped.state.copy()
        root_turn = env.unwrapped.turn
        root = MCTSNode(root_state, root_turn)
        
        for sim in range(num_simulations):
            node = root
            env_copy = gym.make('xiangqi-v0')
            env_copy.reset()
            env_copy.unwrapped.state = root_state.copy()
            env_copy.unwrapped.turn = root_turn
            
            search_path = [node]
            
            while not node.is_leaf():
                action_8100 = node.select(c_puct=self.c_puct)
                env_action = self._convert_action_8100_to_env(action_8100, env_copy)
                env_copy.step(env_action)
                node = node.children[action_8100]
                node.state = env_copy.unwrapped.state.copy()
                node.turn = env_copy.unwrapped.turn
                search_path.append(node)
            
            game_over = self._is_game_over(env_copy)
            
            if not game_over:
                state_enc = self._encode_state(env_copy.unwrapped.state, env_copy.unwrapped.turn)
                state_tensor = torch.from_numpy(state_enc).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(state_tensor)
                    if isinstance(output, tuple):
                        policy_logits, value = output
                    else:
                        policy_logits = output
                        value = torch.zeros(1, 1, device=self.device)
                
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                value = value.cpu().numpy()[0][0]
                
                legal_actions_pairs = self._get_legal_actions_8100(env_copy)
                legal_actions_8100 = [a[0] for a in legal_actions_pairs]
                node.expand(policy, legal_actions_8100)
            else:
                winner = self._get_winner(env_copy)
                if node.turn == 1:
                    value = -winner
                else:
                    value = winner
            
            for i, node in enumerate(reversed(search_path)):
                if i % 2 == 0:
                    node.update(value)
                else:
                    node.update(-value)
            
            env_copy.close()
        
        best_action = None
        best_visits = -1
        
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        
        return best_action


class MCTSEvaluator:
    def __init__(self, model_path, config):
        from src.model.resnet import PolicyValueNet, ResNet10
        self.config = config
        self.device = get_device()
        
        try:
            self.model = PolicyValueNet()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_policy_value = True
        except Exception:
            self.model = ResNet10(num_classes=8100)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_policy_value = False
        
        self.mcts = MCTS(self.model, config)
        self.num_simulations = config.get('mcts', {}).get('num_simulations', 200)
    
    def get_best_move(self, env):
        return self.mcts.run(env, num_simulations=self.num_simulations)
    
    def play_game(self, opponent=None, max_steps=200):
        env = gym.make('xiangqi-v0')
        env.reset()
        moves = []
        
        for step in range(max_steps):
            turn = env.unwrapped.turn
            
            if turn == 1:
                action_8100 = self.get_best_move(env)
            else:
                if opponent:
                    action_8100 = opponent.get_best_move(env)
                else:
                    legal_mask = env.unwrapped.enemy_actions
                    legal_env_actions = np.nonzero(legal_mask)[0]
                    if len(legal_env_actions) == 0:
                        break
                    env_act = int(np.random.choice(legal_env_actions))
                    move_info = action_space_to_move(env_act)
                    piece_id = move_info[0]
                    from_pos = move_info[1]
                    to_pos = move_info[2]
                    from_sq = (9 - from_pos[0]) * 9 + from_pos[1]
                    to_sq = (9 - to_pos[0]) * 9 + to_pos[1]
                    action_8100 = from_sq * 90 + to_sq
            
            moves.append(action_8100)
            
            env_action = self.mcts._convert_action_8100_to_env(action_8100, env)
            _, _, done, _ = env.step(env_action)
            
            if done or self.mcts._is_game_over(env):
                break
        
        winner = self.mcts._get_winner(env)
        env.close()
        return moves, winner


def evaluate_model(model_path, config, num_games=10):
    evaluator = MCTSEvaluator(model_path, config)
    
    results = {'red_wins': 0, 'black_wins': 0, 'draws': 0}
    
    for i in range(num_games):
        _, winner = evaluator.play_game()
        
        if winner == 1:
            results['red_wins'] += 1
        elif winner == -1:
            results['black_wins'] += 1
        else:
            results['draws'] += 1
    
    print(f'Evaluation results over {num_games} games:')
    print(f'Red wins: {results["red_wins"]}')
    print(f'Black wins: {results["black_wins"]}')
    print(f'Draws: {results["draws"]}')
    
    return results


if __name__ == '__main__':
    from src.utils.common import load_config
    config = load_config()
    model_path = config['paths']['models'] + 'resnet10_best.pth'
    evaluate_model(model_path, config)
