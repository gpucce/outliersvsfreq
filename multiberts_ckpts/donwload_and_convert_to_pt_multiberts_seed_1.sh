seed=1

wget "https://storage.googleapis.com/multiberts/public/intermediates/seed_${seed}.zip"
unzip "seed_${seed}.zip"

seed_dir="seed_${seed}"

for i in $(ls $seed_dir)
do
    echo $i
    transformers-cli convert --config bert_config.json --model_type bert --tf_checkpoint $seed_dir/$i/bert.ckpt --pytorch_dump_output $seed_dir/$i/pytorch_model.bin
    cp bert_config.json $seed_dir/$i/config.json
    rm $seed_dir/$i/bert.*
done