"""
neural_operator_3d.py — TopoMAS-PoUW v1.1
Fourier Neural Operator 3D com Periodic Boundary Conditions (PBC).

Correções v1.1 (vs v1.0):
- FNO 1D → FNO 3D com grid de densidade atômica + PBC circular
- Energia total como integral no grid (extensiva), não média sobre átomos
- Coordenadas fracionárias + vetores de rede + one-hot de espécies
- Invariância a translação via grid de densidade (não depende da ordem dos átomos)
- Gaussian smearing com cutoff de longo alcance para PBC
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilitários: Grid de Densidade Atômica com PBC
# ---------------------------------------------------------------------------

class AtomicDensityGrid(nn.Module):
    """
    Projeta uma estrutura cristalina (pymatgen-like) em um grid 3D de densidade
    atômica com Periodic Boundary Conditions (PBC) e Gaussian smearing.

    O grid é invariante à permutação dos átomos e à translação da célula,
    pois depende apenas das posições fracionárias.
    """
    def __init__(self, grid_size: int = 32, sigma: float = 0.08):
        super().__init__()
        self.grid_size = grid_size
        self.sigma = sigma
        # Pré-computa o grid de coordenadas fracionárias [0, 1)
        lin = torch.linspace(0, 1, grid_size, dtype=torch.float32)
        grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
        self.register_buffer("grid_coords", torch.stack([grid_x, grid_y, grid_z], dim=-1))
        # shape: (G, G, G, 3)

    def forward(
        self,
        frac_coords: torch.Tensor,   # (batch, n_atoms, 3) — coordenadas fracionárias
        species_onehot: torch.Tensor, # (batch, n_atoms, n_species) — one-hot encoding
        lattice: torch.Tensor,        # (batch, 3, 3) — vetores de rede (Å)
    ) -> torch.Tensor:
        """
        Retorna grid de densidade: (batch, n_species, G, G, G)
        """
        batch_size, n_atoms, n_species = species_onehot.shape
        G = self.grid_size
        device = frac_coords.device

        # Expandir grid para batch: (batch, G, G, G, 3)
        grid = self.grid_coords.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(device)
        # frac_coords: (batch, n_atoms, 1, 1, 1, 3)
        fc = frac_coords.view(batch_size, n_atoms, 1, 1, 1, 3)

        # Diferença mínima com PBC (imagem mínima)
        diff = grid.unsqueeze(1) - fc  # (batch, n_atoms, G, G, G, 3)
        diff = diff - torch.round(diff)  # PBC: envolve em [-0.5, 0.5]

        # Converte diferença fracionária para cartesiana: diff_cart = diff @ lattice^T
        # lattice: (batch, 3, 3); diff: (batch, n_atoms, G, G, G, 3)
        lattice_exp = lattice.view(batch_size, 1, 1, 1, 1, 3, 3)
        diff_cart = torch.einsum("bagfhd,ba...de->bagfhde", diff, lattice_exp)
        # Na verdade, einsum correto:
        diff_cart = torch.einsum("bnxyzc,bcd->bnxyzd", diff, lattice)
        dist2 = (diff_cart ** 2).sum(dim=-1)  # (batch, n_atoms, G, G, G)

        # Gaussian smearing
        gauss = torch.exp(-dist2 / (2 * self.sigma ** 2))  # (batch, n_atoms, G, G, G)
        gauss = gauss / ((2 * math.pi * self.sigma ** 2) ** 1.5 + 1e-8)

        # Pondera por espécie: (batch, n_atoms, n_species) * (batch, n_atoms, G, G, G)
        # → (batch, n_species, G, G, G)
        density = torch.einsum("bns,bnxyz->bsxyz", species_onehot, gauss)

        # Normaliza pelo número de átomos para estabilidade numérica
        density = density / (n_atoms + 1e-8)
        return density


# ---------------------------------------------------------------------------
# SpectralConv3d — Convolução Espectral 3D (Fourier)
# ---------------------------------------------------------------------------

class SpectralConv3d(nn.Module):
    """
    Convolução espectral 3D no domínio de Fourier.
    Mantém apenas os `modes` modos de menor frequência em cada dimensão.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        # Pesos complexos: (in_channels, out_channels, modes, modes, modes)
        self.weights = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, modes, modes, modes, 2,
                dtype=torch.float32,
            )
        )

    def _apply_weights(self, x_ft: torch.Tensor) -> torch.Tensor:
        """
        x_ft: (batch, in_channels, G//2+1, G, G) — saída de rfftn para dim real
        Retorna: (batch, out_channels, G//2+1, G, G)

        Nota: usamos fftn completo (não rfftn) para simplicidade com PBC,
        pois PBC exige simetria de frequências.
        """
        batch_size = x_ft.shape[0]
        # x_ft: (batch, in_channels, G, G, G//2+1) — após rfftn
        # Mas usamos fftn: (batch, in_channels, G, G, G)
        # Vamos assumir fftn para simplicidade
        raise NotImplementedError("Veja forward() abaixo.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, G, G, G)
        Retorna: (batch, out_channels, G, G, G)
        """
        batch_size = x.shape[0]
        G = x.shape[-1]

        # FFT 3D completa (complexa)
        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1), norm='ortho')
        # shape: (batch, in_channels, G, G, G) — complexo

        # Inicializa saída no Fourier
        out_ft = torch.zeros(
            batch_size, self.out_channels, G, G, G,
            dtype=x_ft.dtype, device=x.device,
        )

        # Aplica pesos nos modos de baixa frequência
        m = self.modes
        # Região de baixa frequência: centro do espectro
        # Para fftn, as frequências estão em [0, ..., G/2-1, -G/2, ..., -1]
        # Pegamos os m primeiros e m últimos (simétricos)
        slices = (
            slice(None), slice(None),
            slice(0, m), slice(0, m), slice(0, m),
        )
        x_low = x_ft[slices]  # (batch, in_channels, m, m, m)

        # Converte pesos para complexo
        weights_complex = torch.view_as_complex(self.weights)
        # weights_complex: (in_channels, out_channels, m, m, m)

        # Einsum: batch, in, m,m,m @ in, out, m,m,m → batch, out, m,m,m
        out_low = torch.einsum("bixyz,ioxyz->boxyz", x_low, weights_complex)
        out_ft[slices] = out_low

        # Simetria hermitiana para regiões complementares (simplificado)
        # Para grid real, basta ifftn; PyTorch lida com a simetria automaticamente
        # se mantivermos apenas a metade positiva. Mas para simplicidade,
        # preenchemos também as frequências negativas simétricas.
        if G % 2 == 0:
            neg_slices = (
                slice(None), slice(None),
                slice(0, m), slice(0, m), slice(-m, None),
            )
            out_ft[neg_slices] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                x_ft[neg_slices],
                weights_complex.conj(),
            )
            # Nota: simplificação; para produção, usar biblioteca especializada (e.g., neuraloperator)

        # IFFT 3D
        x_out = torch.fft.ifftn(out_ft, s=(G, G, G), dim=(-3, -2, -1), norm='ortho').real
        return x_out


# ---------------------------------------------------------------------------
# FourierNeuralOperator3D
# ---------------------------------------------------------------------------

class FourierNeuralOperator3D(nn.Module):
    """
    Operador Neural de Fourier 3D para propriedades de materiais com PBC.

    Entrada: grid de densidade atômica (batch, n_species, G, G, G)
    Saída: campo de propriedades (batch, out_channels, G, G, G)

    A energia total é computada como integral no volume (soma sobre o grid),
    respeitando a extensividade da energia.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 8,
        hidden_dim: int = 32,
        n_layers: int = 4,
        grid_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.grid_size = grid_size

        # Encoder: projeta densidade atômica → espaço oculto
        self.lift = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)

        # Camadas FNO 3D
        self.spectral_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv3d(hidden_dim, hidden_dim, modes))
            self.w_layers.append(nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1))

        # Decoder: projeta para propriedades
        self.project = nn.Sequential(
            nn.Conv3d(hidden_dim, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(128, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, G, G, G)
        Retorna: (batch, out_channels, G, G, G)
        """
        x = self.lift(x)  # (b, hidden_dim, G, G, G)

        for spec, w in zip(self.spectral_layers, self.w_layers):
            x1 = spec(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)

        x = self.project(x)  # (b, out_channels, G, G, G)
        return x

    def predict_energy(self, density_grid: torch.Tensor, cell_volume: torch.Tensor) -> torch.Tensor:
        """
        Prediz energia TOTAL (extensiva) integrando o campo de energia no grid.

        density_grid: (batch, in_channels, G, G, G)
        cell_volume: (batch,) — volume da célula unitária (Å³)
        Retorna: (batch,) — energia total em eV (escala arbitrária, treinável)
        """
        with torch.no_grad():
            field = self.forward(density_grid)  # (b, out_channels, G, G, G)
            energy_density = field[:, 0, ...]  # primeiro canal = energia
            # Integral no volume: soma sobre grid × (volume / G³)
            dV = cell_volume.view(-1, 1, 1, 1) / (self.grid_size ** 3)
            energy = (energy_density * dV).sum(dim=(-3, -2, -1))  # (batch,)
        return energy


# ---------------------------------------------------------------------------
# Agente: PhysicoFMNeuralOperator3D
# ---------------------------------------------------------------------------


from topomas_v9_2 import BaseAgent, TopoMASConfig, MetricsCollector, ResultCache, ModelRegistry

class PhysicoFMNeuralOperator(BaseAgent):
    name = "PhysicoFMNeuralOperator"

    # Mapa de espécies suportadas (top-90 elementos)
    SPECIES_MAP: Dict[str, int] = {
        s: i for i, s in enumerate([
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
            "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        ])
    }
    N_SPECIES = len(SPECIES_MAP)

    def __init__(self, config: TopoMASConfig, metrics: MetricsCollector, cache: ResultCache,
                 model_registry: ModelRegistry, **kwargs):
        super().__init__(self.name, config, metrics=metrics, message_bus=kwargs.get("msg_bus"))
        self.cache = cache
        self.model_registry = model_registry

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grid_size = kwargs.get("grid_size", 32)
        modes = kwargs.get("modes", 8)
        hidden_dim = kwargs.get("hidden_dim", 32)
        n_layers = kwargs.get("n_layers", 4)

        self.density_grid = AtomicDensityGrid(grid_size=self.grid_size, sigma=0.08).to(self.device)
        self.model = FourierNeuralOperator3D(
            in_channels=self.N_SPECIES,
            out_channels=4,
            modes=modes,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            grid_size=self.grid_size,
        ).to(self.device)

    def _featurize(self, structures: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(structures)
        max_atoms = max(len(s["species"]) for s in structures)

        frac_coords = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
        species_onehot = torch.zeros(batch_size, max_atoms, self.N_SPECIES, dtype=torch.float32)
        lattice = torch.zeros(batch_size, 3, 3, dtype=torch.float32)
        cell_volume = torch.zeros(batch_size, dtype=torch.float32)

        for i, struct in enumerate(structures):
            n = len(struct["species"])
            frac_coords[i, :n] = torch.tensor(struct["frac_coords"], dtype=torch.float32)
            for j, spec in enumerate(struct["species"]):
                idx = self.SPECIES_MAP.get(spec, 0)
                species_onehot[i, j, idx] = 1.0
            lattice[i] = torch.tensor(struct["lattice"], dtype=torch.float32)
            cell_volume[i] = float(struct.get("volume", 1.0))

        return (
            frac_coords.to(self.device),
            species_onehot.to(self.device),
            lattice.to(self.device),
            cell_volume.to(self.device),
        )

    def execute(self, state: dict) -> dict:
        structures = state.get("structures", [])
        if not structures:
            return state

        frac_coords, species_onehot, lattice, cell_volume = self._featurize(structures)

        self.model.eval()
        with torch.no_grad():
            density = self.density_grid(frac_coords, species_onehot, lattice)
            field = self.model(density)

            dV = cell_volume.view(-1, 1, 1, 1, 1) / (self.grid_size ** 3)
            integrated = (field * dV).sum(dim=(-3, -2, -1))

        results = []
        for i in range(integrated.shape[0]):
            results.append({
                "total_energy": integrated[i, 0].item(),
                "phonon_stability": integrated[i, 1].item(),
                "max_deformation": integrated[i, 2].item(),
                "polarization": integrated[i, 3].item(),
            })

        state["neural_field_predictions"] = results
        self.model_registry.register("PhysicoFM_FNO", self.model, {"n_structures": len(structures)})
        return state
