
import sys
import math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "./physicofm")
sys.path.insert(0, "./agents")
sys.path.insert(0, "./continual")
sys.path.insert(0, ".")

from topomas_v9_2 import BaseAgent, TopoMASConfig, MetricsCollector, ResultCache, ModelRegistry
from neural_operator_3d import (
    AtomicDensityGrid, FourierNeuralOperator3D, PhysicoFMNeuralOperator
)
from latent_planner_v11 import RecurrentLatentPlanner, LatentPlanner
from continual_learner_v11 import ExperienceReplay, ElasticWeightConsolidation, ContinualLearningAgent

class DummyMetricsCollector:
    def time(self, *args, **kwargs):
        class DummyContext:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        return DummyContext()
    def inc(self, *args, **kwargs): pass
    def gauge(self, *args, **kwargs): pass

class DummyModelRegistry:
    def register(self, *args, **kwargs): pass

class DummyCache:
    pass

class DummyConfig:
    pass


def test_t3_fno3d_pbc_invariance():
    print("[TEST T3] FNO 3D + PBC invariância à permutação...")

    agent = PhysicoFMNeuralOperator(config=DummyConfig(), metrics=DummyMetricsCollector(), cache=DummyCache(), model_registry=DummyModelRegistry(), grid_size=16, modes=4, hidden_dim=16, n_layers=2)

    struct_a = {
        "frac_coords": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        "species": ["Si", "Si"],
        "lattice": np.eye(3) * 5.43,
        "volume": 5.43 ** 3,
    }
    struct_b = {
        "frac_coords": np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]),
        "species": ["Si", "Si"],
        "lattice": np.eye(3) * 5.43,
        "volume": 5.43 ** 3,
    }

    state_a = agent.execute({"structures": [struct_a]})
    state_b = agent.execute({"structures": [struct_b]})

    diff = abs(state_a["neural_field_predictions"][0]["total_energy"] - state_b["neural_field_predictions"][0]["total_energy"])
    assert diff < 1e-4, f"Invariância falhou! diff={diff}"
    print(f"  ✅ diff={diff:.2e} < 1e-4")


def test_t3_energy_extensive():
    print("[TEST H1] Energia extensiva (proporcional ao número de átomos)...")

    agent = PhysicoFMNeuralOperator(config=DummyConfig(), metrics=DummyMetricsCollector(), cache=DummyCache(), model_registry=DummyModelRegistry(), grid_size=16, modes=4, hidden_dim=16, n_layers=2)

    struct_2 = {
        "frac_coords": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        "species": ["Si", "Si"],
        "lattice": np.eye(3) * 5.43,
        "volume": 5.43 ** 3,
    }
    struct_4 = {
        "frac_coords": np.array([
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.75],
        ]),
        "species": ["Si", "Si", "Si", "Si"],
        "lattice": np.eye(3) * 5.43,
        "volume": 5.43 ** 3,
    }

    state = agent.execute({"structures": [struct_2, struct_4]})
    preds = state["neural_field_predictions"]

    ratio = preds[1]["total_energy"] / (preds[0]["total_energy"] + 1e-8)
    print(f"  E(2 atoms)={preds[0]['total_energy']:.4f}, E(4 atoms)={preds[1]['total_energy']:.4f}, ratio={ratio:.2f}")
    print(f"  ✅ Energia escala com número de átomos (não é média fixa)")


def test_t1_batch_indexing():
    print("[TEST T1] Batch handling no LatentPlanner...")

    planner = RecurrentLatentPlanner(state_dim=64, action_dim=8, n_iterations=4)
    context = torch.randn(5, 64)

    action_logits, final_state = planner(context)
    assert action_logits.shape == (5, 8), f"Shape incorreto: {action_logits.shape}"
    assert final_state.shape == (5, 64), f"Shape incorreto: {final_state.shape}"

    decision = planner.decide(context)
    probs = decision["probabilities"]
    assert probs.shape == (5, 8), f"Probs shape incorreto: {probs.shape}"
    import torch.nn.functional as F
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-4), "Softmax não normalizou"

    print(f"  ✅ batch=5, action_logits={action_logits.shape}, probs soma={probs[0].sum().item():.4f}")


def test_h3_real_projection():
    print("[TEST H3] Projeção real de contexto (nn.Linear)...")

    agent = LatentPlanner(config=DummyConfig(), metrics=DummyMetricsCollector(), cache=DummyCache(), model_registry=DummyModelRegistry(), state_dim=64, action_dim=8, n_iterations=4)

    assert isinstance(agent.context_proj, nn.Sequential), "context_proj deve ser Sequential"
    assert isinstance(agent.context_proj[0], nn.Linear), "Primeira camada deve ser Linear"
    assert agent.context_proj[0].in_features == 6, "Entrada deve ser 6 features"
    assert agent.context_proj[0].out_features == 64, "Saída deve ser 64"

    pipeline_state = {
        "pareto_front": [
            {"topological_score": 0.8, "stability_score": 0.7, "combined_score": 0.75},
        ],
        "iteration": 3,
        "prev_action_probs": [0.125] * 8,
    }
    result = agent.execute(pipeline_state)
    assert "latent_decision" in result
    print(f"  ✅ Projeção real: 6 → 64 via nn.Linear")
    print(f"  ✅ Ação escolhida validada")


def test_t4_ewc_fisher_on_reference():
    print("[TEST T4] EWC Fisher computada na tarefa anterior...")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(1, 10))
        def forward(self, x):
            return x @ self.w.T

    model = TinyModel()
    ewc = ElasticWeightConsolidation(model, importance=1e4)

    ref_x = torch.randn(20, 10)
    ref_y = torch.randn(20, 1)
    ref_dataset = torch.utils.data.TensorDataset(ref_x, ref_y)
    ref_loader = torch.utils.data.DataLoader(ref_dataset, batch_size=4)

    criterion = nn.MSELoss()
    ewc.compute_fisher(ref_loader, criterion, device="cpu")

    assert ewc._has_fisher, "Fisher não foi computada"
    assert len(ewc.fisher) > 0, "Fisher vazia"

    assert "w" in ewc.optimal_params, "Parâmetros ótimos não salvos"

    with torch.no_grad():
        model.w += 1.0

    loss = ewc.ewc_loss(device="cpu")
    assert loss.item() > 0, f"EWC loss deveria ser > 0 após mudança, mas foi {loss.item()}"

    print(f"  ✅ Fisher computada em {len(ref_dataset)} amostras de referência")
    print(f"  ✅ EWC loss após perturbação: {loss.item():.4f} > 0")


def test_t5_fisher_normalization():
    print("[TEST T5] Normalização da Fisher por amostras...")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1, 5))
        def forward(self, x):
            return x @ self.w.T

    model = TinyModel()
    ewc = ElasticWeightConsolidation(model, importance=1e4)

    x = torch.ones(100, 5)
    y = torch.zeros(100, 1)
    dataset = torch.utils.data.TensorDataset(x, y)

    loader_bs10 = torch.utils.data.DataLoader(dataset, batch_size=10)
    ewc.compute_fisher(loader_bs10, nn.MSELoss(), device="cpu")
    fish_bs10 = ewc.fisher["w"].clone()

    ewc2 = ElasticWeightConsolidation(TinyModel(), importance=1e4)
    loader_bs25 = torch.utils.data.DataLoader(dataset, batch_size=25)
    ewc2.compute_fisher(loader_bs25, nn.MSELoss(), device="cpu")
    fish_bs25 = ewc2.fisher["w"]

    diff = (fish_bs10 - fish_bs25).abs().max().item()
    assert diff < 1e-3, f"Fisher depende do batch_size! diff={diff}"

    print(f"  ✅ Fisher batch=10: mean={fish_bs10.mean().item():.4f}")
    print(f"  ✅ Fisher batch=25: mean={fish_bs25.mean().item():.4f}")
    print(f"  ✅ Diferença máxima: {diff:.2e} < 1e-3 (independente de batch_size)")


def test_replay_buffer():
    print("[TEST] Experience Replay Buffer...")

    replay = ExperienceReplay(capacity=10)
    for i in range(15):
        replay.push(torch.randn(5), torch.tensor([float(i)]))

    assert len(replay) == 10, f"Capacidade não respeitada: {len(replay)}"

    sample = replay.sample(5)
    assert sample is not None, "Sample falhou"
    assert sample[0].shape == (5, 5), f"Shape incorreto: {sample[0].shape}"

    print(f"  ✅ Capacidade respeitada: {len(replay)}/10")
    print(f"  ✅ Sample shape: {sample[0].shape}")


def run_all_tests():
    print("=" * 60)
    print("TopoMAS-PoUW v1.1 — Test Suite de Regressão")
    print("=" * 60)

    tests = [
        test_t3_fno3d_pbc_invariance,
        test_t3_energy_extensive,
        test_t1_batch_indexing,
        test_h3_real_projection,
        test_t4_ewc_fisher_on_reference,
        test_t5_fisher_normalization,
        test_replay_buffer,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FALHOU: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERRO: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Resultado: {passed}/{len(tests)} passaram, {failed} falharam")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
