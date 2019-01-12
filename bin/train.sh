python trainer.py --model capsule \
 --model_config /notebooks/source/classifynet/model_config.json \
 --model_dir /data/xuht/eventy_detection/sentiment/model/ \
 --config_prefix /notebooks/source/classifynet/configs \
 --gpu_id 0 \
 --train_path "/data/xuht/eventy_detection/sentiment/model/train_10_16_char.txt" \
 --dev_path "/data/xuht/eventy_detection/sentiment/model/test_10_16_char.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/eventy_detection/sentiment/model/idf/emb_mat.pkl" \
 --label_emb_path "/data/xuht/eventy_detection/sentiment/model/label_emb.pkl" \
 --user_dict_path "/data/xuht/eventy_detection/inference_data/project_entity.txt" \
 --emb_idf 0

