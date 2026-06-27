import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.common import load_config
from src.trainer.game_generator import generate_games
from src.trainer.imitation_trainer import ImitationTrainer
from src.trainer.rl_trainer import train_rl
from src.trainer.mcts_evaluator import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Chinese Chess AI Training Pipeline')
    parser.add_argument('--stage', type=str, required=True, 
                        choices=['generate', 'imitation', 'rl', 'evaluate'],
                        help='Training stage to run')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.stage == 'generate':
        print('Starting game generation...')
        generate_games(config)
    
    elif args.stage == 'imitation':
        print('Starting imitation learning...')
        trainer = ImitationTrainer(config)
        trainer.train()
    
    elif args.stage == 'rl':
        print('Starting RL fine-tuning...')
        train_rl(config)
    
    elif args.stage == 'evaluate':
        print('Starting evaluation...')
        import subprocess
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate_models.py'), '--config', args.config]
        subprocess.run(cmd)

if __name__ == '__main__':
    main()