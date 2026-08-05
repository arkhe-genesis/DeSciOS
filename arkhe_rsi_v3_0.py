"""
================================================================================
ARKHE-RSI-v3.0 — Recursao de Interferencia com Ensemble (versao corrigida)
================================================================================
Autor: Arkhe Architect (draft) + correcoes de implementacao
Data:  2026-08-05

IDEIA CENTRAL (correta, mantida do draft):
------------------------------------------
v2.0: rho = outer(M,M)/||M||^2         -> rank 1  -> S(rho) = 0 (congelado)
v3.0: rho = (1/N) sum_k |M_k><M_k|/||M_k||^2 -> rank <= N -> S(rho) > 0 (vivo)

A matriz de densidade e uma MISTURA DE ENSEMBLE sobre amostras estocasticas
de pares (I,J). Isso produz estados MISTOS genuinos, sem regularizacao ad-hoc.
Essa parte do draft esta matematicamente correta e foi preservada.

CORRECOES em relacao ao draft pasted (v3.0 "95/100"):
------------------------------------------------------
1. BUG: I e J eram amostrados com a MESMA chave k5 -> I == J exatamente ->
   fidelity(I,J) == 1.0 sempre. Agora usam chaves independentes.
2. recursion_step reconstruia um RSIState aleatorio inteiro (127x127 * varios)
   so para descarta-lo via .replace(); alem disso nunca atualizava I/J. Agora
   e uma transicao pura e barata que atualiza Lambda e re-amostra (I,J).
3. A alegacao "a entropia EVOLUI" confundia deriva dinamica com ruido de
   amostragem (cada passo usava uma chave nova). Agora medimos as duas coisas
   separadamente: uma trajetoria de chave-FIXA (dinamica pura de Lambda) e o
   desvio-padrao de S sob re-amostragem com Lambda fixo (piso de ruido).
4. Caminho de saida Linux (/mnt/agents/output) -> saida relativa ao script.

DEPENDENCIAS: pip install jax jaxlib matplotlib
EXECUCAO:     python arkhe_rsi_v3_0.py
================================================================================
"""

from functools import partial
import os

import jax
import jax.numpy as jnp
from jax import grad, vmap, random

# matplotlib e opcional: o script imprime todas as verificacoes numericas mesmo
# sem backend grafico. So tentamos plotar se o import funcionar.
try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception as _e:  # pragma: no cover - ambiente sem matplotlib
    _HAVE_MPL = False
    _MPL_ERR = _e

# ==============================================================================
# CONFIGURACAO
# ==============================================================================
DIM = 127          # dimensao do espaco de Hilbert
ENSEMBLE_N = 64    # tamanho do ensemble (controla o rank maximo de rho)
SEED = 42
PERT = 0.5         # escala da perturbacao de (I,J) em torno de Lambda

# ==============================================================================
# ESTADO RSI (pytree registrado — compativel com JAX autodiff)
# ==============================================================================


class RSIState:
    """Estado da recursao RSI, estruturado como pytree JAX.

    Campos-folha (differenciaveis / transformaveis): B, W1, b1, W2, b2, Lambda,
    I, J. `dim` fica em aux_data (estatico), portanto nao vira folha e nao e
    diferenciado nem "tracado".
    """

    __slots__ = ("B", "W1", "b1", "W2", "b2", "Lambda", "I", "J", "dim")

    def __init__(self, dim, key=None):
        if key is None:
            key = random.PRNGKey(SEED)
        k_B, k_W1, k_W2, k_L, k_I, k_J = random.split(key, 6)

        # Operador B: projecao linear.
        self.B = random.normal(k_B, (dim, dim)) * 0.05
        # Operador R: MLP de 2 camadas.
        self.W1 = random.normal(k_W1, (dim, dim)) * 0.1
        self.b1 = jnp.zeros(dim)
        self.W2 = random.normal(k_W2, (dim, dim)) * 0.1
        self.b2 = jnp.zeros(dim)
        # Estado latente Lambda e amostras (I,J) — CHAVES INDEPENDENTES (fix #1).
        self.Lambda = random.normal(k_L, (dim,)) * 0.1
        self.I = self.Lambda + random.normal(k_I, (dim,)) * PERT
        self.J = self.Lambda + random.normal(k_J, (dim,)) * PERT
        self.dim = dim

    # -- pytree protocol -------------------------------------------------------
    def tree_flatten(self):
        children = (self.B, self.W1, self.b1, self.W2, self.b2,
                    self.Lambda, self.I, self.J)
        aux_data = {"dim": self.dim}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.B, obj.W1, obj.b1, obj.W2, obj.b2,
         obj.Lambda, obj.I, obj.J) = children
        obj.dim = aux_data["dim"]
        return obj

    # -- update funcional (imutavel) ------------------------------------------
    def replace(self, **kwargs):
        obj = RSIState.__new__(RSIState)
        obj.B = kwargs.get("B", self.B)
        obj.W1 = kwargs.get("W1", self.W1)
        obj.b1 = kwargs.get("b1", self.b1)
        obj.W2 = kwargs.get("W2", self.W2)
        obj.b2 = kwargs.get("b2", self.b2)
        obj.Lambda = kwargs.get("Lambda", self.Lambda)
        obj.I = kwargs.get("I", self.I)
        obj.J = kwargs.get("J", self.J)
        obj.dim = kwargs.get("dim", self.dim)
        return obj


jax.tree_util.register_pytree_node(
    RSIState,
    lambda s: s.tree_flatten(),
    RSIState.tree_unflatten,
)

# ==============================================================================
# OPERADORES
# ==============================================================================


def R_operator(x, W1, b1, W2, b2):
    """Operador nao-linear R: MLP de 2 camadas com tanh."""
    h = jnp.tanh(W1 @ x + b1)
    return jnp.tanh(W2 @ h + b2)


def membrane(I, J, B, W1, b1, W2, b2):
    """Membrana de interferencia: M(I,J) = R(BI+BJ) - R(BI) - R(BJ).

    Mede a NAO-LINEARIDADE de R: se R fosse linear, M seria identicamente 0.
    """
    BI = B @ I
    BJ = B @ J
    return (R_operator(BI + BJ, W1, b1, W2, b2)
            - R_operator(BI, W1, b1, W2, b2)
            - R_operator(BJ, W1, b1, W2, b2))

# ==============================================================================
# MATRIZ DE DENSIDADE (ENSEMBLE)
# ==============================================================================


def sample_pair(Lambda, key):
    """Amostra (I,J) como perturbacoes INDEPENDENTES de Lambda."""
    k1, k2 = random.split(key)
    I = Lambda + random.normal(k1, Lambda.shape) * PERT
    J = Lambda + random.normal(k2, Lambda.shape) * PERT
    return I, J


def _density_term(Lambda, B, W1, b1, W2, b2, key):
    """Um termo |M><M|/||M||^2 do ensemble."""
    I, J = sample_pair(Lambda, key)
    M = membrane(I, J, B, W1, b1, W2, b2)
    norm_sq = jnp.dot(M, M) + 1e-10
    return jnp.outer(M, M) / norm_sq


@partial(jax.jit, static_argnames=("n_samples",))
def density_matrix(Lambda, B, W1, b1, W2, b2, key, n_samples=ENSEMBLE_N):
    """rho = (1/N) sum_k |M_k><M_k| / ||M_k||^2.

    Mistura de ensemble -> estado misto genuino (rank <= N), permitindo
    S(rho) > 0 sem hacks. Deterministica dada `key` (fix #3).
    """
    keys = random.split(key, n_samples)
    terms = vmap(lambda k: _density_term(Lambda, B, W1, b1, W2, b2, k))(keys)
    rho = jnp.mean(terms, axis=0)
    rho = (rho + rho.T) / 2                     # simetrizacao numerica
    rho = rho / (jnp.trace(rho) + 1e-10)        # normalizacao do traco
    return rho

# ==============================================================================
# METRICAS QUANTICAS
# ==============================================================================


def von_neumann_entropy(rho):
    """S(rho) = -Tr(rho log rho)."""
    eigvals = jnp.linalg.eigvalsh(rho)
    eigvals = jnp.clip(eigvals, 1e-12, 1.0)
    return -jnp.sum(eigvals * jnp.log(eigvals))


def purity(rho):
    """P = Tr(rho^2)."""
    return jnp.trace(rho @ rho)


def fidelity(I, J):
    """Fidelidade de estados (nao-normalizados): |<I|J>|^2 / (||I||^2 ||J||^2)."""
    return jnp.abs(jnp.dot(I, J)) ** 2 / (jnp.dot(I, I) * jnp.dot(J, J) + 1e-10)

# ==============================================================================
# SHARPNESS (OBJETIVO DIFERENCIAVEL)
# ==============================================================================


def sharpness(state, key, n_samples=ENSEMBLE_N):
    """Sharpness = S(rho): entropia de von Neumann da matriz de densidade.

    Diferenciavel em relacao a todos os parametros via JAX autodiff.
    """
    rho = density_matrix(state.Lambda, state.B, state.W1, state.b1,
                         state.W2, state.b2, key, n_samples=n_samples)
    return von_neumann_entropy(rho)


def sharpness_of_W1(W1_flat, state, key):
    """Wrapper: S(rho) como funcao de W1 achatado (para jax.grad)."""
    W1 = W1_flat.reshape((state.dim, state.dim))
    return sharpness(state.replace(W1=W1), key)

# ==============================================================================
# RECURSAO MESTRA (transicao pura e barata — fix #2)
# ==============================================================================


def recursion_step(state, key):
    """U_{n+1} = Phi(U_n + eta * M_bar(U_n)).

    M_bar = membrana media sobre um mini-batch do ensemble. Operadores
    (B, W1, b1, W2, b2) sao FIXOS; atualizamos Lambda com saturacao tanh e
    re-amostramos (I,J) a partir do novo Lambda para que evoluam de fato.
    """
    k_batch, k_pair = random.split(key)
    keys = random.split(k_batch, 32)

    def single_membrane(k):
        I, J = sample_pair(state.Lambda, k)
        return membrane(I, J, state.B, state.W1, state.b1, state.W2, state.b2)

    M_avg = jnp.mean(vmap(single_membrane)(keys), axis=0)
    new_Lambda = jnp.tanh(state.Lambda + 0.1 * M_avg)
    new_I, new_J = sample_pair(new_Lambda, k_pair)
    return state.replace(Lambda=new_Lambda, I=new_I, J=new_J)

# ==============================================================================
# MAIN — DEMONSTRACAO / VERIFICACAO
# ==============================================================================


def main():
    print("=" * 70)
    print("ARKHE-RSI-v3.0 — DEMONSTRACAO JAX (corrigida)")
    print("=" * 70)

    key = random.PRNGKey(SEED)
    state = RSIState(DIM, key)

    # [1] M != 0 (R e de fato nao-linear)
    key, k1 = random.split(key)
    I0, J0 = sample_pair(state.Lambda, k1)
    M = membrane(I0, J0, state.B, state.W1, state.b1, state.W2, state.b2)
    print(f"\n[1] ||M|| = {jnp.linalg.norm(M):.6f}  (nao-zero => R nao-linear)")

    # [2] rho misto: rank > 1, S > 0, P < 1
    key, k = random.split(key)
    rho = density_matrix(state.Lambda, state.B, state.W1, state.b1,
                         state.W2, state.b2, k, n_samples=ENSEMBLE_N)
    eigvals = jnp.linalg.eigvalsh(rho)
    rank = int(jnp.sum(eigvals > 1e-6))
    S = float(von_neumann_entropy(rho))
    P = float(purity(rho))
    print(f"[2] rank(rho) = {rank}  (esperado <= N={ENSEMBLE_N}, e > 1)")
    print(f"    S(rho)     = {S:.6f}  (> 0)")
    print(f"    P=Tr(rho^2)= {P:.6f}  (< 1)")
    print(f"    Tr(rho)    = {float(jnp.trace(rho)):.6f}  (== 1)")
    assert rank > 1 and S > 0.0 and P < 1.0

    # [3] Gradiente via autodiff
    key, k = random.split(key)
    grad_S = grad(sharpness_of_W1)(state.W1.flatten(), state, k)
    gnorm = float(jnp.linalg.norm(grad_S))
    print(f"[3] ||grad_W1 S(rho)|| = {gnorm:.6e}  "
          f"({'nao-nulo => diferenciavel' if gnorm > 0 else 'NULO!'})")
    assert jnp.all(jnp.isfinite(grad_S))

    # [4] Fidelidade agora e nao-trivial (fix #1)
    F0 = float(fidelity(state.I, state.J))
    print(f"[4] fidelity(I,J) inicial = {F0:.6f}  "
          f"(< 1 => I != J, bug do draft corrigido)")
    assert F0 < 0.9999

    # [5] Recursao: dinamica pura (chave FIXA) vs. piso de ruido de amostragem
    print(f"\n[5] Recursao mestra (20 passos):")
    fixed_key = random.PRNGKey(777)          # mesma chave em todos os passos
    S_dyn, P_hist, F_hist = [], [], []
    state_rec = state
    for _ in range(20):
        key, k = random.split(key)
        state_rec = recursion_step(state_rec, k)
        # medida com chave FIXA => variacao reflete so a deriva de Lambda
        rho_n = density_matrix(state_rec.Lambda, state_rec.B, state_rec.W1,
                               state_rec.b1, state_rec.W2, state_rec.b2,
                               fixed_key, n_samples=ENSEMBLE_N)
        S_dyn.append(float(von_neumann_entropy(rho_n)))
        P_hist.append(float(purity(rho_n)))
        F_hist.append(float(fidelity(state_rec.I, state_rec.J)))

    # piso de ruido: S com Lambda FIXO (o inicial) sob re-amostragem
    noise = []
    for _ in range(20):
        key, k = random.split(key)
        rho_n = density_matrix(state.Lambda, state.B, state.W1, state.b1,
                               state.W2, state.b2, k, n_samples=ENSEMBLE_N)
        noise.append(float(von_neumann_entropy(rho_n)))
    noise = jnp.array(noise)
    S_dyn_arr = jnp.array(S_dyn)
    dyn_range = float(jnp.max(S_dyn_arr) - jnp.min(S_dyn_arr))
    noise_std = float(jnp.std(noise))
    print(f"    S dinamico (chave fixa): {S_dyn_arr[0]:.4f} -> {S_dyn_arr[-1]:.4f}"
          f"  (amplitude {dyn_range:.4f})")
    print(f"    piso de ruido de amostragem: std(S) = {noise_std:.4f}")
    verdict = ("deriva de Lambda domina o ruido"
               if dyn_range > 3 * noise_std
               else "variacao comparavel ao ruido de amostragem")
    print(f"    => {verdict}")

    # [6] Figura (se matplotlib disponivel)
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "arkhe_rsi_v3_0_jax.png")
    if not _HAVE_MPL:
        print(f"\n[6] matplotlib indisponivel ({_MPL_ERR!r}); pulando a figura.")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ax = axes[0, 0]
        theta = jnp.linspace(0, 4 * jnp.pi, 500)
        r = jnp.linspace(0.1, 5, 500)
        amp = 1 + 0.3 * jnp.sin(theta * 3 + jnp.linalg.norm(M[:2]) * theta)
        ax.plot(r * jnp.cos(theta) * amp, r * jnp.sin(theta) * amp,
                lw=0.8, alpha=0.7, color="#00d4ff")
        ax.set_title("Arkhe Spiral — amplitude modulada por ||M||")
        ax.set_aspect("equal"); ax.axis("off")

        ax = axes[0, 1]
        ax.plot(S_dyn, marker="o", color="#ff6b6b", lw=2, label="S dinamico")
        ax.axhspan(float(noise.mean() - noise_std),
                   float(noise.mean() + noise_std),
                   color="gray", alpha=0.25, label="+/-1 std ruido")
        ax.set_title("Entropia S(rho) — dinamica vs. ruido")
        ax.set_xlabel("passo n"); ax.set_ylabel("S(rho)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(P_hist, marker="s", color="#4ecdc4", lw=2)
        ax.set_title("Pureza P = Tr(rho^2)")
        ax.set_xlabel("passo n"); ax.set_ylabel("P"); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        sc = ax.scatter(F_hist, S_dyn, c=range(len(F_hist)),
                        cmap="viridis", s=60)
        ax.set_title("Fidelidade vs Entropia (colorido por passo)")
        ax.set_xlabel("fidelity |<I|J>|^2"); ax.set_ylabel("S(rho)")
        ax.grid(True, alpha=0.3); fig.colorbar(sc, ax=ax, label="passo")

        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
        print(f"\n[6] Figura salva em {out_png}")

    print("\n" + "=" * 70)
    print("ARKHE-RSI-v3.0 — COMPLETO (todas as assercoes passaram)")
    print("=" * 70)


if __name__ == "__main__":
    main()
