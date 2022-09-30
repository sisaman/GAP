ARGDIR="$(dirname "$(realpath "$0")")"

mapfile -t <"$ARGDIR/attack.conf"
ATTACK_ARGS=${MAPFILE[@]}

mapfile -t <"$ARGDIR/sage.conf"
SAGE_ARGS=${MAPFILE[@]}

python attack.py sage-inf gra $SAGE_ARGS $ATTACK_ARGS $@
