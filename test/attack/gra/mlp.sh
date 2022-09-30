ARGDIR="$(dirname "$(realpath "$0")")"

mapfile -t <"$ARGDIR/attack.conf"
ATTACK_ARGS=${MAPFILE[@]}

mapfile -t <"$ARGDIR/mlp.conf"
MLP_ARGS=${MAPFILE[@]}

python attack.py mlp gra $MLP_ARGS $ATTACK_ARGS $@