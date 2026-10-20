from core.attacks.gra import GraphReconstructionAttack
from core.attacks.lira import LikelihoodRatioAttack
from core.attacks.lsa import LinkStealingAttack
from core.attacks.nmi import NodeMembershipInference

supported_attacks = {
    'lsa': LinkStealingAttack,
    'nmi': NodeMembershipInference,
    'lira': LikelihoodRatioAttack,
    'gra': GraphReconstructionAttack,
}
