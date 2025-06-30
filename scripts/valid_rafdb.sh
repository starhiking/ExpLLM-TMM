IDX=0,1,2,3

IFS=',' read -r -a array <<< "$IDX"
len_node=${#array[@]}

export PYTHONPATH=$PYTHONPATH:./

output_dir=./checkpoints/ckpts/RAFDB_91.03

if [ -d ${output_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_dir}
fi

output_eval_dir=${output_dir}/eval
if [ -d ${output_eval_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_eval_dir}
fi


CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 /home/lanxing/anaconda3/envs/locLLM/bin/torchrun --nnodes=1 --nproc_per_node=$len_node --master_port=25005 \
    utils/validfaceEMO.py \
    --model-name ${output_dir} \
    --question-file data_list/test/raf-db.txt \
    --output-dir ${output_eval_dir} \
    --conv-format facetask  2>&1 | tee ${output_eval_dir}/eval.txt


python utils/eval_metrics.py --eval_dir ${output_eval_dir} 2>&1 | tee ${output_eval_dir}/metrics.txt

#### you need change path and get a test des file 
# python scripts/get_test_des.py

output_eval_dir=${output_dir}/eval-des
if [ -d ${output_eval_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_eval_dir}
fi

#### question file should change according the path !
CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 /home/lanxing/anaconda3/envs/locLLM/bin/torchrun --nnodes=1 --nproc_per_node=$len_node --master_port=25005 \
    utils/validfaceEMO-des.py \
    --model-name ${output_dir} \
    --question-file data_list/test/raf-db-91.03-des.txt \
    --output-dir ${output_eval_dir} \
    --conv-format facetask  2>&1 | tee ${output_eval_dir}/eval.txt