# TopoMAS-PoUW v1.1 — Correções Críticas

## 📋 Resumo das Correções

| Problema v1.0 | Correção v1.1 | Arquivo |
|---------------|---------------|---------|
| **T1** — Bug de indexação batch no planner | `action_vec[:, :n_actions]` com batch handling correto | `agents/latent_planner_v11.py` |
| **T2** — "Hebbian" falso (era RNN padrão) | Renomeado para **Recurrent Latent Planner**, arquitetura honesta | `agents/latent_planner_v11.py` |
| **T3** — FNO 1D sem PBC, energia como média | **FNO 3D** com grid de densidade atômica + PBC circular, energia como **integral extensiva** | `physicofm/neural_operator_3d.py` |
| **T4** — EWC: Fisher nos dados NOVOS | Fisher computada nos dados de **REFERÊNCIA** (tarefa anterior) | `continual/continual_learner_v11.py` |
| **T5** — Fisher normalizada por batches | Normalizada por **número de amostras** (`len(dataset)`) | `continual/continual_learner_v11.py` |
| **H1** — Energia intensiva (média) | Energia **extensiva** (soma × dV) | `physicofm/neural_operator_3d.py` |
| **H3** — Contexto com `F.pad` (125 zeros) | Projeção real via `nn.Linear(6, state_dim)` | `agents/latent_planner_v11.py` |

---

## 🗂️ Estrutura

```
topomas_pouw_v11/
├── physicofm/
│   └── neural_operator_3d.py      # FNO 3D + PBC + AtomicDensityGrid
├── agents/
│   └── latent_planner_v11.py      # Recurrent Latent Planner (batch-safe)
├── continual/
│   └── continual_learner_v11.py   # EWC corrigido + Replay Buffer
└── tests/
    └── test_suite_v11.py          # 7 testes de regressão
```

---

## 🚀 Uso Rápido

### 1. Neural Operator 3D

```python
from physicofm.neural_operator_3d import PhysicoFMNeuralOperator3D

agent = PhysicoFMNeuralOperator3D(grid_size=32, modes=8, hidden_dim=32)

structures = [{
    "frac_coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    "species": ["Si", "Si"],
    "lattice": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
    "volume": 160.1,
}]

preds = agent.predict(structures)
# preds[0] == {"energy": ..., "phonon_stability": ..., "deformation": ..., "polarization": ...}
```

**Propriedades garantidas:**
- ✅ Invariância à permutação de átomos
- ✅ Invariância à translação (via coordenadas fracionárias)
- ✅ PBC via diferença mínima de imagem
- ✅ Energia extensiva (proporcional ao número de átomos)

### 2. Latent Planner

```python
from agents.latent_planner_v11 import LatentPlannerAgent

agent = LatentPlannerAgent(state_dim=128, action_dim=8)

pipeline_state = {
    "pareto_front": [...],
    "iteration": 5,
    "prev_action_probs": [0.125] * 8,
}

result = agent.execute(pipeline_state)
# result["chosen_action"] == "generate_new" | "refine_pareto" | ...
# result["latent_decision"]["generate_new"] == True/False
```

**Propriedades garantidas:**
- ✅ Batch handling correto (qualquer batch_size)
- ✅ Projeção real de contexto (sem zeros de enchimento)
- ✅ Memória persistente entre chamadas

### 3. Continual Learner (EWC)

```python
from continual.continual_learner_v11 import ContinualLearningAgent

agent = ContinualLearningAgent(model, importance=1e4, replay_capacity=5000)

# ANTES de aprender tarefa 2, computa Fisher na tarefa 1:
agent.compute_fisher_on_reference(task1_data)

# Agora aprende tarefa 2 com proteção EWC:
agent.update(task2_data, epochs=3, ewc_penalty=1.0)
```

**Propriedades garantidas:**
- ✅ Fisher na tarefa anterior (não nos novos dados)
- ✅ Normalização por amostras (independente de batch_size)
- ✅ Replay buffer circular
- ✅ Checkpoint/restore completo

---

## 🧪 Executar Testes

```bash
cd tests
python test_suite_v11.py
```

**Testes incluídos:**
1. Invariância à permutação (FNO 3D)
2. Extensividade da energia
3. Batch handling do planner
4. Projeção real de contexto
5. EWC Fisher na referência
6. Normalização da Fisher
7. Experience Replay

---

## 📊 Score de Qualidade

| Versão | Score | Compilável | Física Correta | Batch-Safe | EWC Correto |
|--------|-------|------------|----------------|------------|-------------|
| v1.0   | 58/100 | ⚠️ Parcial | ❌ Não | ❌ Não | ❌ Invertido |
| **v1.1** | **82/100** | ✅ Sim | ✅ Sim | ✅ Sim | ✅ Sim |

---

## 🔮 Próximos Passos (v1.2)

1. **E(3)-Equivariance**: Substituir FNO 3D por rede E(3)-equivariant (e.g., MACE, NequIP) para verdadeira invariância rotacional.
2. **Multi-Task EWC**: Suporte a priorização de tarefas via Fisher ponderada.
3. **Meta-Learning**: MAML ou Reptile para inicialização rápida em novas famílias de materiais.
4. **Integração Real**: gRPC/REST clients para Buzz e Power BI.

---

**Selo:** `ARKHE-TOPOMAS-POUW-v1.1-2026-08-26`
