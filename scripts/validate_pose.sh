set -x

CONFIG=$1
CKPT=$2
PORT=${3:-12345}
HOST=$(hostname -i)

python ./scripts/validate_pose.py \
    --cfg ${CONFIG} \
    --valid-batch 64 \
    --flip-test \
    --checkpoint ${CKPT} \
    --launcher none \
    --dist-url tcp://${HOST}:${PORT} \
