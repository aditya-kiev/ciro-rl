"""CIRO - Causal Independence Regularization for Offline RL.

A companion repo to the CIRO paper: port of the Conditional Independence
Regularization (CIR) method, originally for contrastive image learning
(causal-cir, github.com/aditya-kiev/causal-cir), to contrastive
state-representation learning for offline RL, plus the ACS and DTG diagnostics.
"""

__version__ = "0.1.0"

from .config import Config, compute_total_steps, load_config  # noqa: F401
from .utils.seeding import default_seeds, set_seed  # noqa: F401

PAPER_ANCHORS = [
    "Algorithm 1 (CIRO)", "Section 5 Definition", "Section 6.1 ACS",
    "Section 6.2 DTG", "Appendix A HSIC", "Appendix B SCM-MDP",
    "Appendix C hyperparameters", "Appendix D averaging",
]