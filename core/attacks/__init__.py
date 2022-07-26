from core.attacks.base import ModelBasedAttack
from core.attacks.lsa import LinkStealingAttack
from core.attacks.nmi import NodeMembershipInference


supported_attacks = {
    'lsa': LinkStealingAttack,
    'nmi': NodeMembershipInference,
}
