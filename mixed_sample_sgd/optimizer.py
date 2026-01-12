import torch
from torch.optim import Optimizer

class MixedSampleSGD(Optimizer):
    r"""
    Implements Mixed-Sample SGD algorithm.
    
    Args:
        params (iterable): 参数组 $\theta$ (Target Model parameters)
        params_q (iterable): 参数组 $\theta_Q$ (Reference/Anchor Model parameters)
        lr (float): 学习率 $\eta$ (for $\theta$ and $\lambda$)
        alpha (float): $\theta_Q$ 的学习率 $\alpha_t$
        gamma (float, optional): $\lambda$ 的衰减系数 $\gamma$ (default: 1e-4)
        epsilon_q (float, optional): 偏差容忍度 $\epsilon_Q$ (default: 0.1)
        weight_decay (float, optional): L2 penalty (default: 0)
    """

    def __init__(self, params, params_q, lr_params=1e-3, lr_lambda=1e-3, alpha=1e-3, gamma=1e-4, epsilon_q=0.1, weight_decay=0):
        if lr_params < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")

        # 将 params 和 params_q 整理成列表
        self.params = list(params)
        self.params_q = list(params_q)
        
        if len(self.params) != len(self.params_q):
            raise ValueError("Params and Params_Q must have the same number of tensors")

        defaults = dict(lr_params=lr_params, lr_lambda=lr_lambda, alpha=alpha, gamma=gamma, epsilon_q=epsilon_q, weight_decay=weight_decay)
        
        # 初始化父类，管理主参数 theta
        super(MixedSampleSGD, self).__init__(self.params, defaults)

        # 内部状态管理
        self.state['lambda_t'] = torch.tensor(0.0) # 初始化 lambda_0 = 0
        
        # 将 params_q 存储在单独的组里，或者直接作为属性管理
        # 为了简单起见，我们假设 params 和 params_q 是一一对应的，直接用索引访问
        # 注意：Optimizer 基类只管理传给它的 self.params (即 theta)。
        # theta_q 我们需要手动应用更新。

    def get_sampling_probability(self):
        """
        Calculates Bernoulli probability p = 1 / (1 + lambda_t)
        Returns:
            bool: True if should sample from S_P (Target), False if from S_Q (Source/Anchor)
        """
        lambda_t = self.state['lambda_t']
        prob = 1.0 / (1.0 + lambda_t.item())
        return torch.bernoulli(torch.tensor(prob)).item() == 1.0

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs the update step for theta (Main Model).
        Equation: theta_{t+1} = theta_t - eta * (1 + lambda_t) * grad(theta_t)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lambda_t = self.state['lambda_t']
        
        for group in self.param_groups:
            lr_params = group['lr_params']
            lr_lambda = group['lr_lambda']
            weight_decay = group['weight_decay']
            
            # 缩放因子 (1 + lambda_t)
            scale = 1.0 + lambda_t

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Update Rule: theta = theta - lr * (1+lambda) * grad
                p.add_(d_p, alpha=-lr_params * scale)

        return loss

    @torch.no_grad()
    def update_verifier(self, loss_theta, loss_theta_q):
        """
        Updates lambda and theta_Q (Reference Model).
        Should be called after computing losses on a batch from S_Q.
        
        Args:
            loss_theta (torch.Tensor): Loss of Main Model on S_Q data -> l(theta_t)
            loss_theta_q (torch.Tensor): Loss of Anchor Model on S_Q data -> l(theta_Q,t)
        """
        # 获取超参
        # 假设所有 param_groups 共享同样的全局超参 (lr, alpha 等)
        group = self.param_groups[0]
        lr_params = group['lr_params']       # eta for theta
        lr_lambda = group['lr_lambda']       # eta for lambda

        alpha = group['alpha'] # alpha_t
        gamma = group['gamma']
        eps_q = group['epsilon_q']
        
        lambda_t = self.state['lambda_t']

        # --- 1. Update Lambda ---
        # Formula: lambda_{t+1} = ReLU[ (1 - gamma*eta)*lambda_t + eta*(l(theta) - l(theta_q) - 6*eps_q) ]
        
        # 确保 loss 是标量
        diff = loss_theta.item() - loss_theta_q.item() -  eps_q
        
        new_lambda = (1 - gamma * lr_lambda) * lambda_t + lr_lambda * diff
        self.state['lambda_t'] = torch.relu(new_lambda) # Projection to [0, inf)

        # --- 2. Update Theta_Q ---
        # Formula: theta_Q,t+1 = theta_Q,t - alpha * grad(theta_Q,t)
        # 注意：这里假设在调用此函数前，用户已经对 theta_q 对应的 loss 执行了 backward()
        
        for i, p_q in enumerate(self.params_q):
            if p_q.grad is None:
                continue
            
            d_p_q = p_q.grad
            # 简单的 SGD 更新
            p_q.add_(d_p_q, alpha=-alpha)
            
            # 更新完后建议清空梯度，防止累积
            p_q.grad = None