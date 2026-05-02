export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CKPT="$1"
MODEL_NAME=$(basename "$CKPT")



# multi3drefer
echo "Evaluating on Multi3DRefer...===================="
ANWSER_FILE="$MODEL_NAME/results/multi3drefer/$MODEL_NAME.jsonl"
python llava/eval/eval_multi3drefer.py --input-file $ANWSER_FILE



#scan2cap
echo "Evaluating on Scan2Cap...===================="
ANWSER_FILE="$MODEL_NAME/results/scan2cap/$MODEL_NAME.jsonl"
# python llava/eval/eval_scan2cap.py --input-file $ANWSER_FILE




#scanqa
echo "Evaluating on ScanQA...===================="
ANWSER_FILE="$MODEL_NAME/results/scanqa/$MODEL_NAME.jsonl"
# python llava/eval/eval_scanqa.py --input-file $ANWSER_FILE



# scanrefer
echo "Evaluating on ScanRefer...===================="
ANWSER_FILE="$MODEL_NAME/results/scanrefer/$MODEL_NAME.jsonl"
python llava/eval/eval_scanrefer.py --input-file $ANWSER_FILE




# sqa3d
ANWSER_FILE="$MODEL_NAME/results/sqa3d/$MODEL_NAME.jsonl"
python llava/eval/eval_sqa3d.py --input-file $ANWSER_FILE
echo "Evaluating on SQA3D...===================="