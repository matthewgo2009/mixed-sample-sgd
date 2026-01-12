import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from dataclasses import dataclass
from sklearn.model_selection import KFold

# === 引入我们的新库 ===
# 确保你已经 pip install -e .
from mixed_sample_sgd import MixedSampleSGD

# ==========================================
# 1. 配置模块 (Configuration)
# ==========================================
@dataclass
class Config:
    data_path: str = "features/cat_dog_resnet18.npz"
    d: int = 512
    
    # Optimization Hyperparameters
    lr: float = 1e-3        # 对应原来的 eta_msgd / alpha
    alpha_q: float = 1e-3   # 对应原来的 alpha (reference update)
    gamma: float = 1e-4     # lambda decay
    epsilon_q: float = 0.1
    batch_size: int = 4
    
    # Experiment Settings
    test_size: int = 1000
    n_Q: int = 100
    epoch_msgd: int = 200   # 适当调整 epoch 以适应 PyTorch 速度
    rep_time: int = 3
    n_P_cand: list = None

    def __post_init__(self):
        if self.n_P_cand is None:
            self.n_P_cand = [100, 300, 500, 700, 900, 1100]
            
    @property
    def folder_name(self):
        return f'results/pytorch_cifar_{self.lr}_{self.epsilon_q}'

# ==========================================
# 2. PyTorch 辅助模型 (Helpers)
# ==========================================
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def evaluate_acc(model, X, Y):
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze()
        predicted_classes = (preds >= 0.5).float()
        acc = (predicted_classes == Y).float().mean()
    return acc.item()

def evaluate_loss(model, X, Y):
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        preds = model(X).squeeze()
        loss = criterion(preds, Y)
    return loss.item()

# ==========================================
# 3. Solvers (使用 PyTorch 重写)
# ==========================================
class Solvers:
    
    @staticmethod
    def train_mixed_sample_sgd(X_P, Y_P, X_Q, Y_Q, cfg, steps):
        """
        使用我们需要测试的 MixedSampleSGD 库
        """
        d = X_P.shape[1]
        model = LogisticRegression(d)
        model_q = LogisticRegression(d) # Anchor model
        
        # 初始化 weights 为 0 (保持和你原来代码逻辑一致)
        with torch.no_grad():
            model.linear.weight.fill_(0)
            model.linear.bias.fill_(0)
            model_q.linear.weight.fill_(0)
            model_q.linear.bias.fill_(0)

        # === 核心：调用你的库 ===
        optimizer = MixedSampleSGD(
            model.parameters(),
            model_q.parameters(),
            lr=cfg.lr,
            alpha=cfg.alpha_q,
            gamma=cfg.gamma,
            epsilon_q=cfg.epsilon_q
        )
        
        criterion = nn.BCELoss()
        
        model.train()
        model_q.train()

        for step in range(steps):
            # --- Phase 1: Update Theta (Main Model) ---
            # 让优化器决定采样源
            use_target = optimizer.get_sampling_probability()
            
            if use_target:
                idx = torch.randint(0, len(X_P), (cfg.batch_size,))
                x, y = X_P[idx], Y_P[idx]
            else:
                idx = torch.randint(0, len(X_Q), (cfg.batch_size,))
                x, y = X_Q[idx], Y_Q[idx]
            
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            # --- Phase 2: Update Verifier (Lambda & Theta_Q) ---
            # 需要在 Source Domain (Sq) 上计算两个模型的 Loss
            idx_q = torch.randint(0, len(X_Q), (cfg.batch_size,))
            x_q_batch, y_q_batch = X_Q[idx_q], Y_Q[idx_q]
            
            # Theta 在 Sq 上的 loss (不需要梯度，只要数值)
            with torch.no_grad():
                loss_theta_sq = criterion(model(x_q_batch).squeeze(), y_q_batch)
            
            # Theta_Q 在 Sq 上的 loss (需要梯度更新 Theta_Q)
            model_q.zero_grad()
            loss_theta_q_sq = criterion(model_q(x_q_batch).squeeze(), y_q_batch)
            loss_theta_q_sq.backward()
            
            # 调用库函数更新
            optimizer.update_verifier(loss_theta_sq, loss_theta_q_sq)

        return model

    @staticmethod
    def train_standard_sgd(X, Y, cfg, steps):
        d = X.shape[1]
        model = LogisticRegression(d)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
        criterion = nn.BCELoss()
        
        model.train()
        for _ in range(steps):
            idx = torch.randint(0, len(X), (cfg.batch_size,))
            x, y = X[idx], Y[idx]
            
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(), y)
            loss.backward()
            optimizer.step()
        return model

# ==========================================
# 4. 实验管理器 (适配 PyTorch Tensor)
# ==========================================
class Experiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.load_data()
        self.results = {
            'risk': [[], [], []], # MSGD, Q, P (简化一下，去掉了 HTL 以精简代码)
            'acc': [[], [], []],
            'time': []
        }

    def load_data(self):
        # 为了让 example 能直接跑，增加 Dummy Data 生成逻辑
        if os.path.exists(self.cfg.data_path):
            data = np.load(self.cfg.data_path)
            X_C, Y_C = data['X_C'], data['Y_C']
            X_D, Y_D = data['X_D'], data['Y_D']
            self.cfg.d = X_C.shape[1]
        else:
            print("Notice: Data file not found. Using generated dummy data.")
            X_C = np.random.randn(5000, self.cfg.d).astype(np.float32)
            Y_C = np.random.randint(0, 2, 5000).astype(np.float32)
            X_D = np.random.randn(5000, self.cfg.d).astype(np.float32) + 0.5
            Y_D = np.random.randint(0, 2, 5000).astype(np.float32)
        
        # === 关键转换：转为 PyTorch Tensor ===
        self.X_C = torch.FloatTensor(X_C)
        self.Y_C = torch.FloatTensor(Y_C)
        self.X_D = torch.FloatTensor(X_D)
        self.Y_D = torch.FloatTensor(Y_D)

    def generate_data(self, n, ratio):
        n_source = int(ratio * n)
        n_target = n - n_source
        
        idx_c = torch.randperm(len(self.X_C))[:n_source]
        idx_d = torch.randperm(len(self.X_D))[:n_target]
        
        X = torch.cat((self.X_C[idx_c], self.X_D[idx_d]), 0)
        Y = torch.cat((self.Y_C[idx_c], self.Y_D[idx_d]), 0)
        return X, Y

    def run_one_trial(self):
        trial_res = {'acc_msgd': [], 'acc_q': [], 'acc_p': []}
        
        # Fixed Target Data
        X_Q, Y_Q = self.generate_data(self.cfg.n_Q, 0.2)
        X_test, Y_test = self.generate_data(1000, 0.2)
        
        # 1. Baseline: Pure Q (Anchor)
        print("  Training Source-Only Baseline...")
        model_q = Solvers.train_standard_sgd(X_Q, Y_Q, self.cfg, steps=2000)
        acc_q = evaluate_acc(model_q, X_test, Y_test)

        for n_P in self.cfg.n_P_cand:
            print(f"  Processing n_P = {n_P}...")
            X_P, Y_P = self.generate_data(n_P, 0.5)
            
            # 2. Mixed Sample SGD (Our Library)
            start = time.time()
            model_msgd = Solvers.train_mixed_sample_sgd(X_P, Y_P, X_Q, Y_Q, self.cfg, steps=n_P*5)
            t_msgd = time.time() - start
            
            # 3. Target Only
            model_p = Solvers.train_standard_sgd(X_P, Y_P, self.cfg, steps=n_P*5)

            # Record
            trial_res['acc_msgd'].append(evaluate_acc(model_msgd, X_test, Y_test))
            trial_res['acc_q'].append(acc_q) # Same for all n_P
            trial_res['acc_p'].append(evaluate_acc(model_p, X_test, Y_test))
            
        return trial_res

    def run_experiment(self):
        for i in range(self.cfg.rep_time):
            print(f"=== Trial {i+1}/{self.cfg.rep_time} ===")
            res = self.run_one_trial()
            self.results['acc'][0].append(res['acc_msgd'])
            self.results['acc'][1].append(res['acc_q'])
            self.results['acc'][2].append(res['acc_p'])

    def plot_results(self):
        os.makedirs(self.cfg.folder_name, exist_ok=True)
        
        acc_mean = [np.mean(x, axis=0) for x in self.results['acc']]
        acc_std = [np.std(x, axis=0) for x in self.results['acc']]
        
        methods = ['Mixed-SGD (Ours)', 'Source-Only', 'Target-Only']
        
        plt.figure(figsize=(8, 6))
        for i, method in enumerate(methods):
            plt.errorbar(self.cfg.n_P_cand, acc_mean[i], yerr=acc_std[i], label=method, capsize=5)
        
        plt.xlabel('Target Samples (n_P)')
        plt.ylabel('Accuracy')
        plt.title('Performance Comparison (PyTorch Implementation)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.cfg.folder_name}/result_plot.png")
        print(f"Plot saved to {self.cfg.folder_name}/result_plot.png")

if __name__ == "__main__":
    cfg = Config()
    exp = Experiment(cfg)
    exp.run_experiment()
    exp.plot_results()