IDX=0,1,2,3,4,5,6,7

IFS=',' read -r -a array <<< "$IDX"
len_node=${#array[@]}

export PYTHONPATH=$PYTHONPATH:./

data_dir=./data/
output_dir=./checkpoints/ckpts/facetask_pretrain_on_ALL_data

output_eval_dir=${output_dir}/eval
if [ -d ${output_eval_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_eval_dir}
fi

CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=$len_node --master_port=25003 \
    utils/validfacedetect.py \
    --model-name ${output_dir} \
    --question-file datasets/tools/training_list/test_facedetect.txt \
    --output-dir ${output_eval_dir} \
    --conv-format facetask  2>&1 | tee ${output_eval_dir}/eval.txt