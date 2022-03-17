for t in 1 2 3
do
for DATA in $1 $2
do
for D in 0 0.2
do
for L in 2
do
for H in 2 4
do
python main.py \
    --dropout ${D} \
    --lr 1e-3 \
    --batch_size 32 \
    --data ${DATA} \
    --ckpt_dir "experiments/${DATA}/layer_${L}_head_${H}_dropout_${D}_${t}" \
    --n_layers ${L} \
    --n_heads ${H} 
done
done
done
done
done