import os
import yaml
import torch
import numpy as np

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config, config_path='config.yaml'):
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    state_dict = torch.load(path, map_location=get_device())
    
    has_backbone = any(k.startswith('backbone.') for k in state_dict.keys())
    model_has_backbone = any(k.startswith('backbone.') for k in model.state_dict().keys())
    
    if has_backbone != model_has_backbone:
        new_state_dict = {}
        for k, v in state_dict.items():
            if has_backbone and not model_has_backbone:
                new_key = k.replace('backbone.', '')
            else:
                new_key = 'backbone.' + k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    return model

PIECE_ID_TO_SYMBOL = {
    1: 'K',
    2: 'A', 3: 'A',
    4: 'B', 5: 'B',
    6: 'N', 7: 'N',
    8: 'R', 9: 'R',
    10: 'C', 11: 'C',
    12: 'P', 13: 'P', 14: 'P', 15: 'P', 16: 'P',
}

PIECE_SYMBOL_TO_CHANNEL = {
    'K': 0, 'A': 1, 'B': 2, 'N': 3, 'R': 4, 'C': 5, 'P': 6,
    'k': 7, 'a': 8, 'b': 9, 'n': 10, 'r': 11, 'c': 12, 'p': 13
}

def piece_id_to_symbol(piece_id_abs):
    return PIECE_ID_TO_SYMBOL.get(piece_id_abs, None)

def encode_board_from_state(state):
    encoded = np.zeros((14, 10, 9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            v = int(state[r, c])
            if v == 0:
                continue
            sym = piece_id_to_symbol(abs(v))
            if sym is None:
                continue
            ch = sym if v > 0 else sym.lower()
            channel = PIECE_SYMBOL_TO_CHANNEL.get(ch)
            if channel is not None:
                encoded[channel, r, c] = 1.0
    return encoded

def flip_perspective(encoded):
    flipped = np.zeros_like(encoded)
    flipped[0:7] = encoded[7:14][:, ::-1, :]
    flipped[7:14] = encoded[0:7][:, ::-1, :]
    return flipped

def encode_board_from_char(board):
    encoded = np.zeros((14, 10, 9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            piece = board[r][c]
            if piece != '.':
                channel = PIECE_SYMBOL_TO_CHANNEL.get(piece)
                if channel is not None:
                    encoded[channel, r, c] = 1.0
    return encoded

def encode_board(board):
    if hasattr(board, 'piece_map'):
        encoded = np.zeros((14, 10, 9), dtype=np.float32)
        piece_map = board.piece_map()
        for pos, piece in piece_map.items():
            row, col = divmod(pos, 9)
            symbol = piece.symbol().upper() if piece.color else piece.symbol()
            channel = PIECE_SYMBOL_TO_CHANNEL.get(symbol)
            if channel is not None:
                encoded[channel, row, col] = 1.0
        return encoded
    elif isinstance(board, np.ndarray):
        return encode_board_from_state(board)
    else:
        return encode_board_from_char(board)

def flip_action(action):
    from_sq = action // 90
    to_sq = action % 90
    from_row, from_col = divmod(from_sq, 9)
    to_row, to_col = divmod(to_sq, 9)
    from_row = 9 - from_row
    to_row = 9 - to_row
    from_sq = from_row * 9 + from_col
    to_sq = to_row * 9 + to_col
    return from_sq * 90 + to_sq

def decode_move(move):
    from_square = move.from_square
    to_square = move.to_square
    return from_square, to_square

def move_to_action(move):
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 90 + to_square

def action_to_move(action):
    from_square = action // 90
    to_square = action % 90
    return from_square, to_square