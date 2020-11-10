from config import SCHEDULER_POL_DECAY_PARAM, SCHEDULER_STEP_PARAM, SCHEDULER_STEP_EPOCHS_PARAM


class LearningRateScheduler:

    """
        Learning Rate Scheduler for monocular depth.
        Policies:
            - Plateau scheduling, according to MonoDepth (Godard et al), but slightly adjusted steps
            - Step scheduling:          new_lr = baselr * SCHEDULER_STEP_PARAM ^ {epoch // step)}
                                            with step = total_epochs / SCHEDULER_STEP_EPOCHS_PARAM
            - Polynomial scheduling:    new_lr = baselr * (1 - iter/maxiter) ^ SCHEDULER_POL_DECAY_PARAM

        For now I will simply put the magic variables for the step and polynomial
        schedulers in config.py
    """

    def __init__(self, apply_scheduling, mode, base_lr, num_epochs, num_iters):
        self.do_scheduling = apply_scheduling
        self.mode = mode

        self.base_lr = base_lr
        self.num_epochs = num_epochs
        self.num_iters = num_iters
        self.total_iters = num_epochs * num_iters

        # Decrease learning rate every how many epochs.
        self.step = max(1, num_epochs // SCHEDULER_STEP_EPOCHS_PARAM)

    def __call__(self, optimizer, epoch, i):
        # No scheduling.
        if not self.do_scheduling:
            return

        # Plateau scheduling.
        if self.mode == 'plateau':
            if 30 <= epoch < 40:
                new_lr = self.base_lr / 2
            elif epoch >= 40:
                new_lr = self.base_lr / 4
            else:
                return

        # Polynomial scheduling.
        elif self.mode == 'polynomial':
            current_iter = epoch * self.num_iters + i
            new_lr = self.base_lr * (1 - (current_iter / self.total_iters) ** SCHEDULER_POL_DECAY_PARAM)

        # Step scheduling.
        elif self.mode == 'step':
            new_lr = self.base_lr * (SCHEDULER_STEP_PARAM ** (epoch // self.step))
        else:
            raise NotImplementedError("{} not a learning rate policy".format(self.mode))

        assert new_lr >= 0
        self._adjust_learning_rate(optimizer, new_lr)

    @staticmethod
    def _adjust_learning_rate(optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
