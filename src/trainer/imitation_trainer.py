import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm
import numpy as np
from src.model.resnet import ResNet10
from src.utils.common import get_device, save_model, load_config
from src.utils.data_utils import RoundDataset

class IndexedRoundDataset(RoundDataset):
    """支持指定索引的RoundDataset（用于验证集和测试集）"""
    def __init__(self, filepath, indices):
        super().__init__(filepath)
        self.selected_indices = self.valid_indices[indices]
    
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        original_idx = self.selected_indices[idx]
        board = self.boards[original_idx]
        action = self.actions[original_idx]
        return torch.from_numpy(board).float(), torch.tensor(action, dtype=torch.long)

class ImitationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self.model = ResNet10().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # 三阶段训练配置
        self.stage_configs = [
            {
                'epochs': 80,
                'batch_size': 64,
                'lr': 0.001,
                'samples_per_epoch': 300000,
                'val_samples_per_epoch': 50000,
                'description': '快速学习阶段1 - 普通对局(第1-6轮)'
            },
            {
                'epochs': 80,
                'batch_size': 64,
                'lr': 0.0005,
                'samples_per_epoch': 300000,
                'val_samples_per_epoch': 50000,
                'description': '快速学习阶段2 - 普通对局(第7-12轮)'
            },
            {
                'epochs': 80,
                'batch_size': 64,
                'lr': 0.0002,
                'samples_per_epoch': 300000,
                'val_samples_per_epoch': 50000,
                'description': '精细调优阶段 - 高质量对局(第13-15轮)'
            },
        ]
        
        self.save_interval = 5
        self.model_path = config['paths']['models']
        self.data_path = config['paths']['games_data']
        
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs('data/jsons', exist_ok=True)
        
        self.train_log = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'time_per_epoch': [],
            'stage': [],
            'data_type': []
        }
        
        self.best_val_acc = 0.0
        
        self._get_data_files()
        
        self._split_validation_files()
    
    def _get_data_files(self):
        """获取普通对局和高质量对局文件列表"""
        all_files = sorted([f for f in os.listdir(self.data_path) 
                           if f.endswith('.npz') and 'index_cache' not in f])
        
        self.normal_files = [f for f in all_files if 'high_quality' not in f]
        self.high_quality_files = [f for f in all_files if 'high_quality' in f]
        
        print(f"普通对局轮次: {len(self.normal_files)}")
        print(f"高质量对局轮次: {len(self.high_quality_files)}")
    
    def _split_validation_files(self):
        """为每个阶段划分对应的验证集（与训练集数据分布一致）"""
        # 阶段1和2使用普通对局的一部分作为验证
        if len(self.normal_files) >= 12:
            # 从普通对局中划分验证集
            val_file = self.normal_files[-1]  # 最后一个普通对局文件作为验证
            self.stage1_val_file = val_file
            self.stage2_val_file = val_file
            self.normal_files_for_train = self.normal_files[:-1]
        else:
            self.stage1_val_file = self.normal_files[0]
            self.stage2_val_file = self.normal_files[-1] if len(self.normal_files) > 1 else self.normal_files[0]
            self.normal_files_for_train = self.normal_files
        
        # 阶段3使用高质量对局的一部分作为验证
        if len(self.high_quality_files) >= 4:
            self.stage3_val_file = self.high_quality_files[-2]  # 倒数第二个高质量文件
            self.test_file = self.high_quality_files[-1]         # 最后一个作为测试
            self.high_quality_files_for_train = self.high_quality_files[:-2]
        else:
            self.stage3_val_file = self.high_quality_files[-1]
            self.test_file = self.high_quality_files[-1]
            self.high_quality_files_for_train = self.high_quality_files[:-1]
        
        # 预划分测试集
        self._prepare_test_set()
        
        print(f"\n验证/测试文件划分:")
        print(f"  阶段1-2验证文件: {self.stage1_val_file}")
        print(f"  阶段3验证文件: {self.stage3_val_file}")
        print(f"  测试文件: {self.test_file}")
    
    def _prepare_test_set(self):
        """准备测试集"""
        filepath = os.path.join(self.data_path, self.test_file)
        with np.load(filepath, mmap_mode='r') as data:
            actions = data['actions']
            valid_mask = (actions >= 0) & (actions < 8100)
            valid_indices = np.where(valid_mask)[0]
        
        np.random.seed(42)
        self.test_indices = np.arange(len(valid_indices))
        np.random.shuffle(self.test_indices)
        
        print(f"  测试集样本数: {len(self.test_indices):,}")
    
    def _create_dataloader(self, files, batch_size, shuffle=True, max_samples=None, seed=None):
        """创建数据加载器，支持采样限制和固定随机种子"""
        datasets = []
        for filename in files:
            filepath = os.path.join(self.data_path, filename)
            dataset = RoundDataset(filepath)
            datasets.append(dataset)
        
        combined_dataset = ConcatDataset(datasets)
        
        if max_samples is not None and max_samples < len(combined_dataset):
            if seed is None:
                seed = np.random.randint(0, 10000)
            np.random.seed(seed)
            sample_indices = np.random.choice(len(combined_dataset), max_samples, replace=False)
            combined_dataset = Subset(combined_dataset, sample_indices)
        
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        ), len(combined_dataset)
    
    def _train_one_epoch(self, train_loader, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for boards, actions in tqdm(train_loader, desc='Training', leave=False):
            boards = boards.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = self.model(boards)
            loss = self.criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * boards.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy
    
    def _validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for boards, actions in tqdm(val_loader, desc='Validating', leave=False):
                boards = boards.to(self.device, non_blocking=True)
                actions = actions.to(self.device, non_blocking=True)
                
                outputs = self.model(boards)
                loss = self.criterion(outputs, actions)
                
                total_loss += loss.item() * boards.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy
    
    def _save_log(self):
        """保存训练日志到JSON文件"""
        log_path = 'data/jsons/imitation_training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_log, f, indent=4, ensure_ascii=False)
    
    def _train_stage(self, stage_num, config, train_files, val_file, optimizer=None):
        """训练单个阶段"""
        import time
        
        print(f"\n{'='*60}")
        print(f"阶段 {stage_num}: {config['description']}")
        print(f"训练数据: {len(train_files)} 个文件")
        print(f"验证数据: {val_file}")
        print(f"配置: epochs={config['epochs']}, batch_size={config['batch_size']}, lr={config['lr']}")
        print(f"训练采样: {config['samples_per_epoch']:,} 个样本/epoch")
        print(f"验证采样: {config['val_samples_per_epoch']:,} 个样本/epoch")
        print('='*60)
        
        # 如果没有传入optimizer，创建新的；否则继续使用之前的optimizer
        if optimizer is None:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['lr'],
                weight_decay=0.0001
            )
        else:
            # 更新学习率而不重置optimizer状态
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['lr']
            print(f"已更新学习率至: {config['lr']}")
        
        # 学习率调度器（每个阶段独立）
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        stage_start_time = time.time()
        
        # 固定验证集采样种子（同一阶段内保持不变）
        val_seed = np.random.randint(0, 10000)
        
        for epoch in range(config['epochs']):
            epoch_start = time.time()
            
            train_loader, train_size = self._create_dataloader(
                train_files, 
                config['batch_size'], 
                shuffle=True,
                max_samples=config['samples_per_epoch']
            )
            
            val_loader, val_size = self._create_dataloader(
                [val_file], 
                config['batch_size'], 
                shuffle=True,
                max_samples=config['val_samples_per_epoch'],
                seed=val_seed  # 固定种子，保持验证集稳定
            )
            
            train_loss, train_acc = self._train_one_epoch(train_loader, optimizer)
            
            val_loss, val_acc = self._validate(val_loader)
            
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            global_epoch = len(self.train_log['epochs']) + 1
            
            self.train_log['epochs'].append(global_epoch)
            self.train_log['train_loss'].append(train_loss)
            self.train_log['train_acc'].append(train_acc)
            self.train_log['val_loss'].append(val_loss)
            self.train_log['val_acc'].append(val_acc)
            self.train_log['lr'].append(scheduler.get_last_lr()[0])
            self.train_log['time_per_epoch'].append(epoch_time)
            self.train_log['stage'].append(stage_num)
            self.train_log['data_type'].append('normal' if stage_num <= 2 else 'high_quality')
            
            print(f"\n阶段 {stage_num} - Epoch [{epoch+1}/{config['epochs']}] (全局: {global_epoch})")
            print(f"耗时: {epoch_time:.2f}s ({epoch_time/60:.1f}分钟)")
            print(f"学习率: {scheduler.get_last_lr()[0]:.6f}")
            print(f"训练样本数: {train_size:,}")
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            print("---")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_model(self.model, os.path.join(self.model_path, 'resnet10_best.pth'))
                print(f"新最佳模型保存，验证准确率: {val_acc:.4f}")
            
            if global_epoch % self.save_interval == 0:
                model_name = f'resnet10_epoch_{global_epoch}.pth'
                save_model(self.model, os.path.join(self.model_path, model_name))
                print(f"模型 {model_name} 已保存")
            
            self._save_log()
            
            elapsed_time = time.time() - stage_start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = config['epochs'] - (epoch + 1)
            remaining_time = avg_time_per_epoch * remaining_epochs
            print(f"阶段剩余时间: {remaining_time/60:.1f} 分钟 ({remaining_time/3600:.1f}小时)")
        
        stage_total_time = time.time() - stage_start_time
        print(f"\n阶段 {stage_num} 完成!")
        print(f"阶段耗时: {stage_total_time/60:.1f} 分钟 ({stage_total_time/3600:.1f}小时)")
        print(f"阶段最佳验证准确率: {self.best_val_acc:.4f}")
        
        # 返回optimizer供下一阶段使用
        return optimizer
    
    def _final_test(self):
        """最终测试（使用预留的测试集）"""
        print(f"\n{'='*60}")
        print("=== 最终测试 ===")
        print('='*60)
        
        test_loader, test_size = self._create_dataloader([self.test_file], 64, shuffle=False)
        
        print(f"测试集来源: {self.test_file}")
        print(f"测试样本数: {test_size:,}")
        
        test_loss, test_acc = self._validate(test_loader)
        
        print(f"\n📊 最终测试结果:")
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        print('='*60)
        
        self.train_log['final_test'] = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_samples': test_size,
            'test_file': self.test_file
        }
        self._save_log()
    
    def train(self):
        """开始三阶段训练"""
        print(f"=== 开始三阶段模仿学习训练 ===")
        print(f"设备: {self.device}")
        print(f"普通对局文件数: {len(self.normal_files)}")
        print(f"高质量对局文件数: {len(self.high_quality_files)}")
        print(f"总阶段数: {len(self.stage_configs)}")
        
        # 阶段1：普通对局第1-5轮
        stage1_files = self.normal_files_for_train[:5]
        print(f"\n阶段1 使用文件: {stage1_files}")
        
        # 阶段2：普通对局第6-11轮（排除验证文件）
        stage2_files = self.normal_files_for_train[5:]
        print(f"阶段2 使用文件: {stage2_files}")
        
        # 阶段3：高质量对局（排除验证和测试文件）
        stage3_files = self.high_quality_files_for_train
        print(f"阶段3 使用文件: {stage3_files}")
        
        # 开始训练，传递optimizer保持状态
        optimizer = None
        
        optimizer = self._train_stage(
            stage_num=1,
            config=self.stage_configs[0],
            train_files=stage1_files,
            val_file=self.stage1_val_file,
            optimizer=optimizer
        )
        
        optimizer = self._train_stage(
            stage_num=2,
            config=self.stage_configs[1],
            train_files=stage2_files,
            val_file=self.stage2_val_file,
            optimizer=optimizer
        )
        
        optimizer = self._train_stage(
            stage_num=3,
            config=self.stage_configs[2],
            train_files=stage3_files,
            val_file=self.stage3_val_file,
            optimizer=optimizer
        )
        
        self._final_test()
        
        final_model_name = 'resnet10_final.pth'
        save_model(self.model, os.path.join(self.model_path, final_model_name))
        
        print(f"\n{'='*60}")
        print("=== 三阶段模仿学习训练完成 ===")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"最终模型已保存: {final_model_name}")
        print('='*60)
        
        self._save_log()

if __name__ == '__main__':
    config = load_config()
    trainer = ImitationTrainer(config)
    trainer.train()