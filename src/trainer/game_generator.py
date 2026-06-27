import os
import subprocess
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
from src.utils.common import encode_board_from_char, flip_perspective, flip_action

_local_engine = None
_current_max_moves = 250

def init_engine(engine_path, time_limit, hash_size, max_moves=250):
    global _local_engine
    global _current_max_moves
    _local_engine = PikafishEngine(engine_path, time_limit, hash_size)
    _local_engine.start()
    _current_max_moves = max_moves

class PikafishEngine:
    def __init__(self, engine_path, time_limit=0.05, hash_size=512):
        self.engine_path = engine_path
        self.time_limit = time_limit
        self.hash_size = hash_size
        self.process = None
    
    def start(self):
        self.process = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        self.send_command('uci')
        self._read_until('uciok')
        self.send_command(f'setoption name Hash value {self.hash_size}')
    
    def send_command(self, command):
        if self.process:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
    
    def _read_line(self):
        if self.process:
            return self.process.stdout.readline()
    
    def _read_until(self, expected):
        while True:
            line = self._read_line()
            if line:
                if expected in line:
                    break
    
    def get_best_move(self, fen):
        self.send_command(f'position fen {fen}')
        self.send_command(f'go movetime {int(self.time_limit * 1000)}')
        
        best_move = None
        while True:
            line = self._read_line()
            if line:
                if 'bestmove' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        best_move = parts[1]
                    break
        
        return best_move
    
    def stop(self):
        if self.process:
            try:
                self.send_command('quit')
                self.process.wait(timeout=5)
            except:
                self.process.kill()

def parse_fen(fen):
    board = [['.' for _ in range(9)] for _ in range(10)]
    parts = fen.split()
    rows = parts[0].split('/')
    
    for r, row_str in enumerate(rows):
        c = 0
        i = 0
        while i < len(row_str):
            char = row_str[i]
            if char.isdigit():
                empty = int(char)
                for _ in range(empty):
                    board[r][c] = '.'
                    c += 1
                i += 1
            else:
                board[r][c] = char
                c += 1
                i += 1
    
    return board

def board_to_fen(board):
    rows = []
    for r in range(10):
        row_str = ''
        empty = 0
        for c in range(9):
            piece = board[r][c]
            if piece == '.':
                empty += 1
            else:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += piece
        if empty > 0:
            row_str += str(empty)
        rows.append(row_str)
    
    return '/'.join(rows)

def uci_to_move(move_str):
    from_col = ord(move_str[0]) - ord('a')
    from_row = 9 - int(move_str[1])
    to_col = ord(move_str[2]) - ord('a')
    to_row = 9 - int(move_str[3])
    return (from_row, from_col, to_row, to_col)

def uci_to_action(move_str):
    if len(move_str) != 4:
        return None
    
    from_col = ord(move_str[0]) - ord('a')
    from_row = 9 - int(move_str[1])
    to_col = ord(move_str[2]) - ord('a')
    to_row = 9 - int(move_str[3])
    
    from_square = from_row * 9 + from_col
    to_square = to_row * 9 + to_col
    
    return from_square * 90 + to_square


def generate_single_game(args):
    game_idx, max_moves, time_limit = args
    global _local_engine
    try:
        engine = _local_engine
        
        initial_fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
        board = parse_fen(initial_fen)
        game_data = []
        turn = 'w'
        
        for _ in range(max_moves):
            current_fen = board_to_fen(board) + f' {turn} - - 0 1'
            encoded_board = encode_board_from_char(board)
            
            move_str = engine.get_best_move(current_fen)
            
            if not move_str or move_str == '(none)' or len(move_str) != 4:
                break
            
            action = uci_to_action(move_str)
            if action is None:
                break
            
            if turn == 'w':
                game_data.append((encoded_board, action))
            else:
                flipped_board = flip_perspective(encoded_board)
                flipped_action = flip_action(action)
                game_data.append((flipped_board, flipped_action))
            
            fr, fc, tr, tc = uci_to_move(move_str)
            
            board[tr][tc] = board[fr][fc]
            board[fr][fc] = '.'
            
            turn = 'b' if turn == 'w' else 'w'
            
            has_red_king = False
            has_black_king = False
            for r in range(10):
                for c in range(9):
                    if board[r][c] == 'K':
                        has_red_king = True
                    if board[r][c] == 'k':
                        has_black_king = True
            
            if not has_red_king or not has_black_king:
                break
        
        return game_data
    except Exception as e:
        print(f'Error generating game {game_idx}: {e}')
        import traceback
        traceback.print_exc()
        return None

def batch_save_to_disk(boards_list, actions_list, round_idx, games_dir, prefix='games'):
    if not boards_list:
        return 0, 0
    
    all_boards = np.concatenate(boards_list, axis=0)
    all_actions = np.concatenate(actions_list, axis=0)
    
    filename = f'{prefix}_round_{round_idx + 1}.npz'
    filepath = os.path.join(games_dir, filename)
    np.savez_compressed(filepath, boards=all_boards, actions=all_actions)
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    return len(all_boards), file_size

def generate_round(config, round_num, mode_config):
    engine_path = config['paths']['pikafish']
    num_processes = config['game']['num_processes']
    hash_size = config['game']['hash_size']
    time_limit = mode_config['time_limit_per_move']
    max_moves = mode_config['max_moves']
    games_per_round = config['game']['games_per_round']
    games_dir = config['paths']['games_data']
    batch_size = config['game']['batch_size']
    file_prefix = mode_config['file_prefix']
    
    round_start_time = time.time()
    print(f'\n=== Round {round_num} [{file_prefix}] ===')
    print(f'Mode: {file_prefix}, Time per move: {time_limit}s, Max moves: {max_moves}')
    
    all_boards = []
    all_actions = []
    games_completed = 0
    batches_completed = 0
    
    total_batches = (games_per_round + batch_size - 1) // batch_size
    
    with Pool(processes=num_processes, 
              initializer=init_engine,
              initargs=(engine_path, time_limit, hash_size)) as pool:
        
        while games_completed < games_per_round:
            batch_start_idx = games_completed
            batch_end_idx = min(games_completed + batch_size, games_per_round)
            batch_size_actual = batch_end_idx - batch_start_idx
            
            game_indices = list(range(batch_start_idx + (round_num - 1) * games_per_round, 
                                     batch_end_idx + (round_num - 1) * games_per_round))
            
            print(f'Processing batch {batches_completed + 1}/{total_batches} ({batch_size_actual} games)...')
            batch_start = time.time()
            
            args_list = [(idx, max_moves, time_limit) for idx in game_indices]
            results = list(tqdm(
                pool.imap(generate_single_game, args_list),
                total=batch_size_actual,
                desc=f'Batch {batches_completed + 1}'
            ))
            
            valid_games = [game for game in results if game and len(game) > 10]
            
            if valid_games:
                boards_batch = []
                actions_batch = []
                for game in valid_games:
                    for board, action in game:
                        boards_batch.append(board)
                        actions_batch.append(action)
                
                if boards_batch:
                    all_boards.append(np.array(boards_batch, dtype=np.float16))
                    all_actions.append(np.array(actions_batch, dtype=np.int32))
            
            games_completed += batch_size_actual
            batches_completed += 1
            
            batch_time = time.time() - batch_start
            print(f'Batch {batches_completed} done in {batch_time:.2f}s')
    
    moves_count, file_size = batch_save_to_disk(all_boards, all_actions, round_num - 1, games_dir, file_prefix)
    
    round_time = time.time() - round_start_time
    
    print(f'\nRound {round_num} complete!')
    print(f'Saved {games_completed} games ({moves_count} moves)')
    print(f'File size: {file_size:.2f} MB')
    print(f'Round time: {round_time/60:.2f} minutes ({games_completed/round_time:.2f} games/sec)')
    
    return moves_count, file_size, round_time

def generate_games(config):
    games_dir = config['paths']['games_data']
    normal_rounds = config['game']['normal_mode']['rounds']
    high_quality_rounds = config['game']['high_quality_mode']['rounds']
    
    os.makedirs(games_dir, exist_ok=True)
    
    total_start_time = time.time()
    total_moves = 0
    total_file_size = 0
    
    print(f'=== Game Generation Settings ===')
    print(f'Normal mode rounds: {normal_rounds}')
    print(f'High quality mode rounds: {high_quality_rounds}')
    print(f'================================')
    
    all_rounds = sorted(normal_rounds + high_quality_rounds)
    
    for round_num in all_rounds:
        if round_num in normal_rounds:
            mode_config = config['game']['normal_mode']
        else:
            mode_config = config['game']['high_quality_mode']
        
        moves_count, file_size, round_time = generate_round(config, round_num, mode_config)
        
        total_moves += moves_count
        total_file_size += file_size
        
        completed_rounds = all_rounds.index(round_num) + 1
        total_rounds = len(all_rounds)
        estimated_total_time = (time.time() - total_start_time) / completed_rounds * total_rounds
        remaining_time = estimated_total_time - (time.time() - total_start_time)
        print(f'Estimated remaining time: {remaining_time/60:.1f} minutes')
    
    total_time = time.time() - total_start_time
    print(f'\n=== Game generation complete! ===')
    print(f'Total time: {total_time/60:.1f} minutes')
    print(f'Total moves: {total_moves}')
    print(f'Total file size: {total_file_size/1024:.2f} GB')

if __name__ == '__main__':
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    generate_games(config)