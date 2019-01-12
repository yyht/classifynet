python test.py --model gru \
 --model_config /notebooks/source/simnet/model_config.json \
 --model_dir /data/xuht/test/simnet \
 --config_prefix /notebooks/source/simnet/configs \
 --gpu_id 1 \
 --test_path "/data/xuht/duplicate_sentence/LCQMC/test.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/duplicate_sentence/LCQMC/emb_mat.pkl" \
 --model_str "gru_1535100414_1.754304933482093_0.7987499975345351"

