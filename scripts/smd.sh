python -m tsad.main --dataset SMD  --data ./data/SMD/  --valid_prop 0.2  --dropout 0.2 --epochs 50 --hidden_dim 512 --history_w 100 --batch_size 50 --seed 42 --z_dim 4 --model_type tfvae --with_phi_dense --with_theta_dense --num_workers 4 --gpu 0 --with_pyro