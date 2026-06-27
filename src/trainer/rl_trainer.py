import os
import json
import gym
import gym_xiangqi
import numpy as np
from gym_xiangqi.constants import RIVER_LOW, RIVER_HIGH
from gym_xiangqi.utils import action_space_to_move, move_to_action_space
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch
import torch.nn as nn
from src.utils.common import load_config, get_device, ensure_dir, encode_board_from_state, flip_perspective, piece_id_to_symbol, PIECE_SYMBOL_TO_CHANNEL
from src.model.resnet import ResNet10


class XiangqiObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(14, 10, 9),
            dtype=np.float32
        )
    
    def _encode_obs(self):
        state = self.env.unwrapped.state
        turn = self.env.unwrapped.turn
        
        encoded = encode_board_from_state(state)
        
        if turn != 1:
            encoded = flip_perspective(encoded)
        
        return encoded
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._encode_obs()
    
    def step(self, action):
        result = self.env.step(action)
        reward = result[1]
        done = result[2]
        info = result[3] if len(result) > 3 else {}
        return self._encode_obs(), reward, done, info


class XiangqiActionWrapper(gym.Wrapper):
    """将 8100 动作空间 (from_sq*90+to_sq) 映射到环境的 129600 动作空间。
    处理视角翻转：当轮到黑方走时，先将动作坐标翻转到环境坐标系。
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8100)
    
    def _convert_action(self, action):
        from_sq = action // 90
        to_sq = action % 90
        from_row, from_col = divmod(from_sq, 9)
        to_row, to_col = divmod(to_sq, 9)
        
        state = self.env.unwrapped.state
        turn = self.env.unwrapped.turn
        
        if turn == 1:
            from_row_env = from_row
            to_row_env = to_row
            legal_mask = self.env.unwrapped.ally_actions
        else:
            from_row_env = 9 - from_row
            to_row_env = 9 - to_row
            legal_mask = self.env.unwrapped.enemy_actions
        
        piece_val = int(state[from_row_env, from_col])
        piece_id = abs(piece_val)
        
        if piece_id == 0:
            legal_actions = np.nonzero(legal_mask)[0]
            if len(legal_actions) > 0:
                return int(legal_actions[0])
            return 0
        
        env_action = move_to_action_space(piece_id, (from_row_env, from_col), (to_row_env, to_col))
        if 0 <= env_action < len(legal_mask) and legal_mask[env_action]:
            return int(env_action)
        
        legal_actions = np.nonzero(legal_mask)[0]
        if len(legal_actions) > 0:
            return int(legal_actions[0])
        return 0
    
    def step(self, action):
        env_action = self._convert_action(action)
        return self.env.step(env_action)
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result


class XiangqiRewardWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.rewards = config['rewards']
        self.move_count = 0
        self.previous_pieces = None
        self.repetition_count = 0
        self.previous_states = []
        self.max_history = 10

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.move_count = 0
        self.previous_pieces = self._count_pieces()
        self.repetition_count = 0
        self.previous_states = []
        return result

    def _count_pieces(self):
        state = getattr(self.env.unwrapped, 'state', None)
        pieces = {}
        if state is None:
            return pieces
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                v = int(state[r, c])
                if v == 0:
                    continue
                piece_type = abs(v)
                color = 0 if v > 0 else 1
                key = (piece_type, color)
                pieces[key] = pieces.get(key, 0) + 1
        return pieces

    def _get_piece_value(self, piece_type, color=None, pos=None):
        if piece_type == 1:
            return self.rewards.get('king', 1000.0)
        if piece_type in (2, 3):
            return self.rewards.get('advisor', 2.0)
        if piece_type in (4, 5):
            return self.rewards.get('elephant', 2.0)
        if piece_type in (6, 7):
            return self.rewards.get('knight', 4.0)
        if piece_type in (8, 9):
            return self.rewards.get('rook', 9.0)
        if piece_type in (10, 11):
            return self.rewards.get('cannon', 4.5)
        if piece_type in range(12, 17):
            if pos is not None and self._is_pawn_crossed(pos, color):
                return self.rewards.get('pawn_crossed', 2.0)
            return self.rewards.get('pawn', 1.0)
        return 0

    def _is_pawn_crossed(self, pos, color):
        row = pos // 9
        if color == 0:
            return row <= RIVER_LOW
        return row >= RIVER_HIGH

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.move_count += 1

        new_pieces = self._count_pieces()
        captured_value = 0

        for key in self.previous_pieces:
            if key not in new_pieces or new_pieces[key] < self.previous_pieces[key]:
                piece_type, color = key
                captured_value += self._get_piece_value(piece_type, color)

        state = getattr(self.env.unwrapped, 'state', None)
        turn = getattr(self.env.unwrapped, 'turn', None)
        current_state = self._state_to_fen(state, turn) if state is not None else None
        if current_state in self.previous_states:
            self.repetition_count += 1
        else:
            self.repetition_count = 0
        self.previous_states.append(current_state)
        if len(self.previous_states) > self.max_history:
            self.previous_states.pop(0)

        repetition_penalty = 0
        if self.repetition_count >= 3:
            repetition_penalty = self.rewards.get('repetition_penalty', -50.0)

        early_threshold = self.rewards.get('early_move_threshold', 50)
        move_penalty = self.rewards.get('early_move_penalty', -0.1) if self.move_count <= early_threshold else self.rewards.get('late_move_penalty', -0.5)

        total_reward = captured_value + repetition_penalty + move_penalty + reward

        self.previous_pieces = new_pieces

        return obs, total_reward, done, info

    def _state_to_fen(self, state, turn):
        rows = []
        for r in range(state.shape[0]):
            empty = 0
            row_str = ''
            for c in range(state.shape[1]):
                v = int(state[r, c])
                if v == 0:
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    id_abs = abs(v)
                    sym = piece_id_to_symbol(id_abs)
                    if sym is None:
                        sym = '?'
                    sym = sym if v > 0 else sym.lower()
                    row_str += sym
            if empty > 0:
                row_str += str(empty)
            rows.append(row_str)
        side = 'w' if turn == 1 else 'b'
        return '/'.join(rows) + ' ' + side


class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    """使用 ResNet10 作为特征提取器，与模仿学习模型架构一致"""
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.resnet = ResNet10(num_blocks=[1, 1, 1, 1], num_classes=features_dim)
        self._features_dim = features_dim
    
    def forward(self, observations):
        return self.resnet(observations)


def load_pretrained_resnet(feature_extractor, pretrained_path, device):
    print(f"加载模仿学习预训练权重: {pretrained_path}")
    pretrained_state = torch.load(pretrained_path, map_location=device)
    
    if isinstance(pretrained_state, dict) and any(key.startswith('module.') for key in pretrained_state):
        pretrained_state = {key.replace('module.', ''): value for key, value in pretrained_state.items()}
    
    resnet_state = feature_extractor.resnet.state_dict()
    
    loaded_count = 0
    skipped_count = 0
    new_state_dict = {}
    
    for key in resnet_state:
        pretrained_key = key
        if key.startswith('backbone.') and not any(k.startswith('backbone.') for k in pretrained_state):
            pretrained_key = key.replace('backbone.', '')
        
        if pretrained_key in pretrained_state:
            if resnet_state[key].shape == pretrained_state[pretrained_key].shape:
                new_state_dict[key] = pretrained_state[pretrained_key]
                loaded_count += 1
            else:
                new_state_dict[key] = resnet_state[key]
                skipped_count += 1
                print(f"  跳过 {key}: 形状不匹配 {pretrained_state[pretrained_key].shape} vs {resnet_state[key].shape}")
        else:
            new_state_dict[key] = resnet_state[key]
            skipped_count += 1
    
    feature_extractor.resnet.load_state_dict(new_state_dict)
    print(f"成功加载 {loaded_count} 层参数，跳过 {skipped_count} 层")
    return loaded_count, skipped_count


class MetricsCallback(BaseCallback):
    """每 N 步将训练指标保存到 JSON 文件。"""

    def __init__(self, save_freq, log_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.log_path = log_path
        self.metrics = []

    def _on_step(self):
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            entry = {'timestep': self.num_timesteps}
            if len(self.model.ep_info_buffer) > 0:
                entry['ep_rew_mean'] = float(np.mean([ep['r'] for ep in self.model.ep_info_buffer]))
                entry['ep_len_mean'] = float(np.mean([ep['l'] for ep in self.model.ep_info_buffer]))
            for key, value in self.model.logger.name_to_value.items():
                entry[key] = float(value)
            self.metrics.append(entry)
            ensure_dir(os.path.dirname(self.log_path))
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=4, ensure_ascii=False)
            if self.verbose:
                print(f"Metrics saved at timestep {self.num_timesteps}")
        return True


class BestModelCallback(BaseCallback):
    """根据平均回合奖励保存最优模型。"""

    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = float(np.mean([ep['r'] for ep in self.model.ep_info_buffer]))
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                if self.verbose:
                    print(f"New best model saved (mean reward: {mean_reward:.2f})")
        return True


def make_env(config):
    """创建完整包装的环境"""
    env = gym.make('xiangqi-v0')
    env = XiangqiObsWrapper(env)
    env = XiangqiActionWrapper(env)
    env = XiangqiRewardWrapper(env, config)
    return env


def train_rl(config, pretrained_path=None):
    ensure_dir(config['paths']['models'])
    ensure_dir('data/jsons')

    env = make_env(config)

    device = get_device()
    print(f"Using device: {device}")
    
    policy_kwargs = dict(
        features_extractor_class=ResNetFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=float(config['rl']['learning_rate']),
        n_steps=int(config['rl']['n_steps']),
        batch_size=int(config['rl']['batch_size']),
        n_epochs=int(config['rl']['n_epochs']),
        gamma=float(config['rl']['gamma']),
        gae_lambda=float(config['rl']['gae_lambda']),
        ent_coef=float(config['rl']['ent_coef']),
        device=device
    )
    
    if pretrained_path is None:
        pretrained_path = os.path.join(config['paths']['models'], 'resnet10_best.pth')
    
    if os.path.exists(pretrained_path):
        load_pretrained_resnet(model.policy.features_extractor, pretrained_path, device)
    else:
        print(f"警告: 未找到预训练模型 {pretrained_path}，将从头开始训练")

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=config['paths']['models'],
        name_prefix='ppo_xiangqi_resnet'
    )

    metrics_callback = MetricsCallback(
        save_freq=1000,
        log_path='data/jsons/rl_training_log_resnet.json',
        verbose=1
    )

    best_model_callback = BestModelCallback(
        save_path=os.path.join(config['paths']['models'], 'ppo_xiangqi_resnet_best'),
        verbose=1
    )

    model.learn(
        total_timesteps=config['rl']['total_timesteps'],
        callback=[checkpoint_callback, metrics_callback, best_model_callback]
    )

    model.save(os.path.join(config['paths']['models'], 'ppo_xiangqi_resnet_final'))
    print('RL training complete!')


def estimate_training_time(config, num_steps=1000):
    """估算训练所需时间"""
    import time
    
    device = get_device()
    print(f"使用设备: {device}")
    
    env = make_env(config)
    
    policy_kwargs = dict(
        features_extractor_class=ResNetFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        learning_rate=float(config['rl']['learning_rate']),
        n_steps=int(config['rl']['n_steps']),
        batch_size=int(config['rl']['batch_size']),
        n_epochs=int(config['rl']['n_epochs']),
        gamma=float(config['rl']['gamma']),
        gae_lambda=float(config['rl']['gae_lambda']),
        ent_coef=float(config['rl']['ent_coef']),
        device=device
    )
    
    pretrained_path = os.path.join(config['paths']['models'], 'resnet10_best.pth')
    if os.path.exists(pretrained_path):
        load_pretrained_resnet(model.policy.features_extractor, pretrained_path, device)
    
    print(f"\n测试 {num_steps} 步训练时间...")
    start_time = time.time()
    
    model.learn(total_timesteps=num_steps)
    
    elapsed = time.time() - start_time
    steps_per_second = num_steps / elapsed
    
    total_timesteps = config['rl']['total_timesteps']
    estimated_seconds = total_timesteps / steps_per_second
    
    print(f"\n=== 时间估算 ===")
    print(f"测试步数: {num_steps}")
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"速度: {steps_per_second:.2f} 步/秒")
    print(f"总训练步数: {total_timesteps:,}")
    print(f"预计总时间:")
    print(f"  - 秒: {estimated_seconds:.0f}")
    print(f"  - 分钟: {estimated_seconds / 60:.1f}")
    print(f"  - 小时: {estimated_seconds / 3600:.2f}")
    print(f"  - 天: {estimated_seconds / 86400:.2f}")
    
    env.close()
    return estimated_seconds


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'estimate'])
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained imitation model')
    parser.add_argument('--test-steps', type=int, default=1000, help='Number of steps for time estimation')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode == 'estimate':
        estimate_training_time(config, num_steps=args.test_steps)
    else:
        train_rl(config, pretrained_path=args.pretrained)
