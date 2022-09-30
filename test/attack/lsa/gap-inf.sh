ARGDIR="$(dirname "$(realpath "$0")")"

mapfile -t <"$ARGDIR/attack.conf"
ATTACK_ARGS=${MAPFILE[@]}

mapfile -t <"$ARGDIR/gap.conf"
GAP_ARGS=${MAPFILE[@]}

python attack.py gap-inf lsa $GAP_ARGS $ATTACK_ARGS $@