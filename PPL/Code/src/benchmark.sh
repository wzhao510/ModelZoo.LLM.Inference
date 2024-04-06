if [ -n "$MPI_LOCALRANKID" ]; then
    MPI_LOCALRANKID=$MPI_LOCALRANKID
elif [ -n "$OMPI_COMM_WORLD_RANK" ]; then
    MPI_LOCALRANKID=$OMPI_COMM_WORLD_RANK
elif [ -n "$PMI_RANK" ]; then
    MPI_LOCALRANKID=$PMI_RANK
else
    echo "[WARNING] MPI_LOCALRANKID not found, set to 0"
    MPI_LOCALRANKID=0
fi
DEVICE_ID=$MPI_LOCALRANKID
 
STEP=$1
if [ ! -n "$STEP" ]; then
    STEP=0
fi

MODEL=$2
MODEL_PATH="${MODEL}/model_slice_${MPI_LOCALRANKID}/model.onnx"
OUTPUT=$3
OUTPUT_DIR="${OUTPUT}/rank_${MPI_LOCALRANKID}" # we should make the rank_* directories first
if [ ! -d "${OUTPUT_DIR}" ]; then
	mkdir -p "${OUTPUT_DIR}"
fi
DATA=$4
TEST_DATA_DIR="${DATA}/rank_${MPI_LOCALRANKID}"
 
# we should rearrange the input tensors if the model exporting parameters has been changed.
TOKEN_IDS=`ls ${TEST_DATA_DIR}/step${STEP}_token_ids-*`
ATTN_MASK=`ls ${TEST_DATA_DIR}/step${STEP}_attn_mask-*`
SEQSTARTS=`ls ${TEST_DATA_DIR}/step${STEP}_seqstarts-*`
KVSTARTS=`ls ${TEST_DATA_DIR}/step${STEP}_kvstarts-*`
CACHESTARTS=`ls ${TEST_DATA_DIR}/step${STEP}_cachestarts-*`
DECODING_BATCHES=`ls ${TEST_DATA_DIR}/step${STEP}_decoding_batches-*`
START_POS=`ls ${TEST_DATA_DIR}/step${STEP}_start_pos-*`
MAX_SEQLEN=`ls ${TEST_DATA_DIR}/step${STEP}_max_seqlen-*`
MAX_KVLEN=`ls ${TEST_DATA_DIR}/step${STEP}_max_kvlen-*`
KV_CAHCE=`ls ${TEST_DATA_DIR}/step${STEP}_kv_cache-*`
KV_SCALE=`ls ${TEST_DATA_DIR}/step${STEP}_kv_scale-*`
 
TEST_INPUTS="$TOKEN_IDS,$ATTN_MASK,$SEQSTARTS,$KVSTARTS,$CACHESTARTS,$DECODING_BATCHES,$START_POS,$MAX_SEQLEN,$MAX_KVLEN,$KV_CAHCE,$KV_SCALE"
INPUT_DEVICES="device,device,device,device,device,host,device,host,host,device,device"

PPL_SERVER_DIR=$5
CMD="${PPL_SERVER_DIR}/pplnn_llm --use-llm-cuda \
--onnx-model $MODEL_PATH \
--shaped-input-files $TEST_INPUTS \
--save-outputs \
--device-id $DEVICE_ID \
--save-data-dir $OUTPUT_DIR \
--in-devices $INPUT_DEVICES \
--enable-profiling \
--min-profiling-seconds 3 \
--warmup-iterations 10"
 
echo "RUN RANK${MPI_LOCALRANKID} STEP${STEP} -> $CMD"
 
eval "$CMD"

python "$(dirname "$0")/compare.py" ${STEP} ${OUTPUT_DIR} ${TEST_DATA_DIR}



