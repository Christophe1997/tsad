python -m tsad.main --rnn_type GRU --dataset KPI --data ./data/KPI  --valid_prop 0.4  --dropout 0.2 --epochs 100 --hidden_dim 64 --history_w 64 --predict_w 0 --device cpu --batch_size 32 --seed 7777 --emb_dim 32
