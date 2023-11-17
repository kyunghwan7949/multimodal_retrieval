############ RSTPReid ###########


######### Graph Pretrain ############
python train_tokengt.py \
--name iira \
--img_aug \
--batch_size 440 \
--MLM \
--loss_names 'mlm' \
--dataset_name 'RSTPReid' \
--root_dir '/data/data2/IRRA' \
--num_epoch 70 \
--vocab_size 2858 \
--encoder-embed-dim 1024 \
--encoder-layers 12 \
--encoder-attention-heads 16 \
--encoder-ffn-embed-dim 4096 \
--dropout 0.1 \
--attention-dropout 0.1 \
--act-dropout 0.0

