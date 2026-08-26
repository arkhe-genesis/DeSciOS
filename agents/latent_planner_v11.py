"""
latent_planner_v11.py — TopoMAS-PoUW v1.1
Recurrent Latent Planner (não-Hebbiano, batch-safe, projeção real).

Correções v1.1 (vs v1.0):
- Bug de indexação batch corrigido: action_vec[:, :n_actions] em vez de action_vec[:3]
- Projeção real via nn.Linear(3, state_dim) em vez de F.pad com zeros
- Nomeação honesta: "Recurrent Latent", não "Hebbian Associative Memory"
- Batch handling correto em todas as operações
- Gating com projeção separada (não reutiliza input_proj)
- Decisão retorna probabilidades por batch, não apenas primeira amostra
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RecurrentLatentPlanner — RNN com gating e memória persistente
# ---------------------------------------------------------------------------

class RecurrentLatentPlanner(nn.Module):
    """
    Planejador baseado em memória recorrente latente com gating.

    Mantém um estado interno que é refinado por iterações recorrentes.
    Não gera tokens de linguagem — opera diretamente no espaço latente.

    Arquitetura:
      - Projeção de contexto (entrada) → estado
      - Iterações recorrentes com skip-connection espectral + gating
      - Projeção de saída → vetor de ação

    Args:
        state_dim: Dimensão do espaço de estado latente.
        action_dim: Dimensão do espaço de ação (número de ações discretas).
        n_iterations: Número de passos de refinamento recorrente.
        dropout: Probabilidade de dropout para regularização.
    """
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 8,
        n_iterations: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_iterations = n_iterations

        # Projeção de entrada: concat[context, state] → novo estado
        self.input_proj = nn.Linear(state_dim + state_dim, state_dim)

        # Transformação recorrente: estado → estado (via MLP leve)
        self.recurrent_transform = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 2, state_dim),
        )

        # Gating para misturar contexto com estado
        self.gate_proj = nn.Linear(state_dim * 2, state_dim)

        # Normalização de camada (estabiliza recorrência profunda)
        self.norm = nn.LayerNorm(state_dim)

        # Projeção de saída: estado → ação
        self.output_proj = nn.Linear(state_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        context: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executa o planejamento recorrente latente.

        Args:
            context: (batch, state_dim) — embedding do estado atual do pipeline.
            initial_state: (batch, state_dim) ou None — estado persistente anterior.

        Returns:
            action: (batch, action_dim) — logits de ação.
            final_state: (batch, state_dim) — estado latente final (para persistência).
        """
        batch_size = context.size(0)
        device = context.device

        if initial_state is None:
            state = torch.zeros(batch_size, self.state_dim, device=device)
        else:
            state = initial_state

        # Iterações de refinamento recorrente
        for _ in range(self.n_iterations):
            # 1. Transformação recorrente do estado
            delta = self.recurrent_transform(state)
            state = state + delta  # skip-connection

            # 2. Gating com contexto
            gate_input = torch.cat([context, state], dim=-1)  # (batch, 2*state_dim)
            gate = torch.sigmoid(self.gate_proj(gate_input))  # (batch, state_dim)
            state = gate * state + (1.0 - gate) * context

            # 3. Normalização
            state = self.norm(state)

        # Decodifica ação
        action_logits = self.output_proj(state)  # (batch, action_dim)
        return action_logits, state

    def decide(self, context: torch.Tensor, temperature: float = 1.0) -> Dict[str, Any]:
        """
        Retorna decisão legível com probabilidades.

        Args:
            context: (batch, state_dim)
            temperature: Fator de temperatura para softmax (1.0 = padrão).

        Returns:
            Dict com:
              - action_logits: (batch, action_dim)
              - probabilities: (batch, action_dim) — softmax normalizado
              - action_idx: (batch,) — índice da ação escolhida (argmax)
        """
        action_logits, _ = self.forward(context)
        probs = F.softmax(action_logits / temperature, dim=-1)  # (batch, action_dim)
        action_idx = torch.argmax(probs, dim=-1)  # (batch,)
        return {
            "action_logits": action_logits,
            "probabilities": probs,
            "action_idx": action_idx,
        }


# ---------------------------------------------------------------------------
# Agente: LatentPlannerAgent
# ---------------------------------------------------------------------------


from topomas_v9_2 import BaseAgent, TopoMASConfig, MetricsCollector, ResultCache, ModelRegistry

class LatentPlanner(BaseAgent):
    name = "LatentPlanner"
    ACTION_NAMES = [
        "generate_new", "refine_pareto", "run_dft", "expand_search",
        "reduce_uncertainty", "exploit_best", "diversity_push", "stop",
    ]

    def __init__(self, config: TopoMASConfig, metrics: MetricsCollector, cache: ResultCache,
                 model_registry: ModelRegistry, **kwargs):
        super().__init__(self.name, config, metrics=metrics, message_bus=kwargs.get("msg_bus"))
        self.cache = cache
        self.model_registry = model_registry

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_dim = kwargs.get("state_dim", 128)
        self.action_dim = kwargs.get("action_dim", 8)

        self.planner = RecurrentLatentPlanner(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            n_iterations=kwargs.get("n_iterations", 10),
        ).to(self.device)

        self.memory_state: Optional[torch.Tensor] = None

        self.context_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, self.state_dim),
        ).to(self.device)

    def _encode_context(self, pipeline_state: Dict[str, Any]) -> torch.Tensor:
        pareto = pipeline_state.get("pareto_front", [])
        if pareto:
            topo_scores = [p.get("topological_score", 0.0) for p in pareto]
            stab_scores = [p.get("stability_score", 0.0) for p in pareto]
            best_score = max(p.get("combined_score", 0.0) for p in pareto)
            mean_topo = float(np.mean(topo_scores))
            mean_stab = float(np.mean(stab_scores))
            n_pareto = float(len(pareto))
        else:
            mean_topo = 0.0
            mean_stab = 0.0
            n_pareto = 0.0
            best_score = 0.0

        prev_probs = pipeline_state.get("prev_action_probs", [0.125] * self.action_dim)
        entropy = -sum(p * math.log(p + 1e-8) for p in prev_probs)
        iteration = float(pipeline_state.get("iteration", 0)) / 100.0  # normalizado

        features = torch.tensor(
            [mean_topo, mean_stab, n_pareto, entropy, best_score, iteration],
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, 6)

        with torch.no_grad():
            context = self.context_proj(features.to(self.device))  # (1, state_dim)
        return context

    def execute(self, state: dict) -> dict:
        with self.metrics.time("agent_duration_seconds", {"agent": self.name}):
            context = self._encode_context(state)

            mem = self.memory_state
            if mem is not None and mem.device != self.device:
                mem = mem.to(self.device)

            with torch.no_grad():
                action_logits, new_state = self.planner(context, initial_state=mem)
                probs = F.softmax(action_logits, dim=-1).squeeze(0)  # (action_dim,)

            if self.memory_state is None:
                self.memory_state = new_state
            else:
                self.memory_state = 0.9 * self.memory_state + 0.1 * new_state

            action_idx = int(torch.argmax(probs).item())
            chosen_action = self.ACTION_NAMES[action_idx]
            action_probs = probs.cpu().tolist()

            decision = {
                "generate_new": action_probs[0] > 0.25,
                "refine_pareto": action_probs[1] > 0.20,
                "run_dft": action_probs[2] > 0.30,
                "expand_search": action_probs[3] > 0.35,
                "reduce_uncertainty": action_probs[4] > 0.30,
                "exploit_best": action_probs[5] > 0.40,
                "diversity_push": action_probs[6] > 0.25,
                "stop": action_probs[7] > 0.60,
            }

            if decision["generate_new"]:
                n_structures = int(3 + 7 * action_probs[0])
                decision["n_structures"] = n_structures

            state["latent_decision"] = decision
            self.logger.info(f"Decisão latente decodificada: {decision}")
            return state
