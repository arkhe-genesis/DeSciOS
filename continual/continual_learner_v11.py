
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import random

class ExperienceReplay:
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, target):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, target)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return None
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.stack([x[0] for x in batch])
        targets = torch.stack([x[1] for x in batch])
        return states, targets

    def __len__(self):
        return len(self.buffer)

class ElasticWeightConsolidation:
    def __init__(self, model: nn.Module, importance: float = 1e4):
        self.model = model
        self.importance = importance
        self.fisher = {}
        self.optimal_params = {}
        self._has_fisher = False

    def compute_fisher(
        self,
        reference_loader: DataLoader,
        criterion,
        device: str = "cpu",
    ) -> dict:
        self.model.eval()
        fisher_new = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_new[name] = torch.zeros_like(param.data)

        for inputs, labels in reference_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            self.model.zero_grad()
            loss.backward(retain_graph=False)

            batch_size_actual = inputs.size(0)
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                # E[g_i²] ≈ B * E[(grad_mean)²]  (correção por amostras)
                grad_sq = param.grad.detach().clone() ** 2 * batch_size_actual
                fisher_new[name] += grad_sq

        n_samples = max(1, len(reference_loader.dataset))
        for name in fisher_new:
            fisher_new[name] /= n_samples

        self.fisher = fisher_new
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self._has_fisher = True
        return self.fisher

    def ewc_loss(self, device: str = "cpu") -> torch.Tensor:
        loss = torch.tensor(0.0, device=device)
        for name, param in self.model.named_parameters():
            if name in self.fisher and param.requires_grad:
                fisher_val = self.fisher[name].to(device)
                opt_val = self.optimal_params[name].to(device)
                loss += (fisher_val * (param - opt_val).pow(2)).sum()
        return self.importance * loss


from topomas_v9_2 import BaseAgent, TopoMASConfig, MetricsCollector, ResultCache, ModelRegistry

class ContinualLearningAgent(BaseAgent):
    name = "ContinualLearner"

    def __init__(self, config: TopoMASConfig, target_model: nn.Module, metrics: MetricsCollector,
                 cache: ResultCache, model_registry: ModelRegistry, **kwargs):
        super().__init__(self.name, config, metrics=metrics, message_bus=kwargs.get("msg_bus"))
        self.cache = cache
        self.model_registry = model_registry

        self.learner = ElasticWeightConsolidation(target_model, importance=kwargs.get("importance", 1e3))
        self.target_model = target_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = optim.Adam(self.target_model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ExperienceReplay(capacity=kwargs.get("replay_capacity", 5000))

    def execute(self, state: dict) -> dict:
        with self.metrics.time("agent_duration_seconds", {"agent": self.name}):
            new_data = state.get("verified_dft_results") # Espera dados validados

            if not new_data:
                self.metrics.inc("cl_skips_total", 1)
                return {"continual_update": "skipped_no_data"}

            try:
                X = torch.stack([torch.tensor(d["features"]) if not isinstance(d["features"], torch.Tensor) else d["features"] for d in new_data]).float()
                y = torch.stack([torch.tensor(d["target"]) if not isinstance(d["target"], torch.Tensor) else d["target"] for d in new_data]).float()
                loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
            except Exception as e:
                self.logger.error(f"Falha ao criar DataLoader contínuo: {e}")
                return {"continual_update": "error"}

            self.learner.compute_fisher(loader, self.criterion, self.device)

            self.target_model.train()
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                preds = self.target_model(batch_x)
                task_loss = self.criterion(preds, batch_y)
                ewc_penalty = self.learner.ewc_loss()

                (task_loss + ewc_penalty).backward()
                self.optimizer.step()
                total_loss += task_loss.item()

            self.model_registry.register("PhysicoFM_FNO_Continual", self.target_model,
                                          {"loss": total_loss/len(loader), "n_samples": len(new_data)})

            self.learner.optimal_params = {n: p.clone().detach() for n, p in self.target_model.named_parameters() if p.requires_grad}

            self.metrics.gauge("cl_last_loss", total_loss/len(loader))
            self.logger.info(f"EWC Update concluído. Loss: {total_loss/len(loader):.4f}")
            return {"continual_update": "success", "loss": total_loss/len(loader)}
