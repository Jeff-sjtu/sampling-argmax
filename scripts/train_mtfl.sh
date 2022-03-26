set -x

CONFIG=$1
EXPID=${2:-"sampling"}
PORT=${3:-23456}
HOST=$(hostname -i)

python ./scripts/train_mtfl.py \
    --nThreads 20 \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --exp-id ${EXPID} \
    --cfg ${CONFIG}
