import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        self.model.loss_history = [self.model.compute_loss()]
        for i in range(self.max_iter):
            step = self._step()
            current_loss = self.model.compute_loss()
            self.model.loss_history.append(current_loss)
            if np.linalg.norm(step) ** 2 < self.tolerance:
                break
            if np.any(np.isnan(step)):
                break
        


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_train, y_train)
        self.model.w -= learning_rate * gradient
        return -learning_rate * gradient


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        num_objects = self.model.X_train.shape[0]
        batch_indices = np.random.choice(num_objects, size=self.batch_size, replace=False)
        X_batch = self.model.X_train[batch_indices]
        y_batch = self.model.y_train[batch_indices]
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_batch, y_batch)
        self.model.w -= learning_rate * gradient
        return -learning_rate * gradient


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)
        indices = np.random.choice(num_objects, size=self.batch_size, replace=False)

        for idx in indices:
            new_grad = self.model.compute_gradients(X_train[idx:idx+1], y_train[idx:idx+1]).flatten()
            old_grad = self.grad_memory[idx]
            self.grad_sum += new_grad - old_grad
            self.grad_memory[idx] = new_grad

        avg_grad = self.grad_sum / num_objects
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        self.model.w -= learning_rate * avg_grad
        return -learning_rate * avg_grad


        


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        learning_rate = self.lr_schedule.get_lr(self.iteration)
        grad = self.model.compute_gradients()
        self.velocity = self.beta * self.velocity + learning_rate * grad
        self.model.w -= self.velocity
        return -self.velocity

class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)
        grad = self.model.compute_gradients()
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        t = self.iteration + 1
        m = self.m / (1 - self.beta1 ** t)
        v = self.v / (1 - self.beta2 ** t)
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        self.model.w -= learning_rate * m / (np.sqrt(v) + self.eps)
        delta = -learning_rate * m / (np.sqrt(v) + self.eps)
        return delta


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        X = self.model.X_train
        y = self.model.y_train
        w = self.model.loss_function.analytic_solution(X, y)
        self.model.w = w
        self.model.loss_history = [self.model.compute_loss(X, y)]
