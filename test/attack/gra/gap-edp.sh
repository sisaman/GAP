ARGDIR="$(dirname "$(realpath "$0")")"

mapfile -t <"$ARGDIR/attack.conf"
ATTACK_ARGS=${MAPFILE[@]}

mapfile -t <"$ARGDIR/gap.conf"
GAP_ARGS=${MAPFILE[@]}

for EPSILON in 8 4 2 1 0.5 0.1
do
    python attack.py gap-edp gra --shadow_epsilon $EPSILON $GAP_ARGS $ATTACK_ARGS $@
done
