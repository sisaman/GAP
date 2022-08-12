from core.attacks.lsa import LinkStealingAttack
from core.attacks.nmi import NodeMembershipInference
from core.attacks.lira import LikelihoodRatioAttack


supported_attacks = {
    'lsa': LinkStealingAttack,
    'nmi': NodeMembershipInference,
    'lira': LikelihoodRatioAttack
}
