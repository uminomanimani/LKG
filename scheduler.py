class CustomLRScheduler:
    def __init__(self, optimizer, warmup_steps, maintain_steps=0, decay_factor=0.98, min_lr=0.0):
        """
        自定义学习率调度器。
        
        参数：
        - optimizer: 优化器。
        - warmup_steps: Warm-up 步数。
        - maintain_steps: 维持学习率的步数，在 Warm-up 完成后保持最大学习率的步数。
        - decay_factor: 学习率衰减系数，默认为 0.98。
        - min_lr: 学习率下限，默认为 0.0，可以是标量或列表。
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.maintain_steps = maintain_steps
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.current_step = 0
        self.warmup_done = False
        self.maintain_step_count = 0
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_metric = None

        # 将学习率初始化为 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.0

    def step(self):
        """
        在训练过程中每个训练步骤调用，用于 Warm-up 和维持阶段。
        """
        if not self.warmup_done:
            # 处于 Warm-up 阶段
            self.current_step += 1
            if self.current_step <= self.warmup_steps:
                # 线性增加学习率
                for idx, param_group in enumerate(self.optimizer.param_groups):
                    initial_lr = self.initial_lrs[idx]
                    new_lr = initial_lr * self.current_step / self.warmup_steps
                    param_group['lr'] = new_lr
            else:
                # Warm-up 完成，将学习率设为 initial_lr
                self.warmup_done = True
                self.maintain_step_count = 0  # 进入维持阶段，重置维持步骤计数
                for idx, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.initial_lrs[idx]
        else:
            # Warm-up 已完成
            if self.maintain_step_count < self.maintain_steps:
                # 仍处于维持阶段，保持学习率不变
                self.maintain_step_count += 1
            # 如果维持阶段也已经完成，不做任何操作，等待衰减条件满足

    def decay_if_warmup_and_maintein_finished(self, metric):
        """
        在验证集指标（如 F1）检查后调用，根据指标调整学习率。
        
        - 如果 Warm-up 未完成或维持阶段未结束，则调用此方法无效。
        - 在维持阶段结束后，如果指标没有提升，则按照衰减因子衰减学习率。
        """
        # Warm-up 或维持阶段尚未结束，调用无效
        if not self.warmup_done or self.maintain_step_count < self.maintain_steps:
            return

        # Warm-up 和维持阶段都已经结束，开始根据指标衰减学习率
        if self.last_metric is not None and metric <= self.last_metric:
            # 如果指标没有提升，衰减学习率
            for idx, param_group in enumerate(self.optimizer.param_groups):
                current_lr = param_group['lr']
                new_lr = current_lr * self.decay_factor

                # 确保学习率不低于 min_lr
                if isinstance(self.min_lr, list):
                    min_lr_value = self.min_lr[idx]
                else:
                    min_lr_value = self.min_lr

                param_group['lr'] = max(new_lr, min_lr_value)

        # 更新 last_metric
        self.last_metric = metric

    def get_current_lr(self):
        """
        获取当前的学习率。
        
        返回：
        - 一个列表，包含每个参数组的学习率。
        """
        return [group['lr'] for group in self.optimizer.param_groups]


def adjustable_lr_scheduler(optimizer, warmup_steps, maintain_steps=0, decay_factor=0.98, min_lr=0.0):
    """
    创建自定义的学习率调度器。
    
    参数：
    - optimizer: 优化器。
    - warmup_steps: Warm-up 步数。
    - maintain_steps: 在 Warm-up 完成后维持最大学习率的步数。
    - decay_factor: 学习率衰减系数。
    - min_lr: 学习率下限，默认为 0.0，可以是标量或列表。
    
    返回：
    - 一个自定义的学习率调度器对象。
    """
    return CustomLRScheduler(optimizer, warmup_steps, maintain_steps, decay_factor, min_lr)
