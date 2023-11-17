############ COCO ###########


######### Finetuning ############
python train_graph.py \
--name iira \
--img_aug \
--batch_size 32 \
--MLM \
--loss_names 'sdm+mlm' \
--dataset_name 'COCO' \
--root_dir '/data/data2/IRRA' \
--num_epoch 65 \
--vocab_size 26964 \
--encoder-embed-dim 1024 \
--encoder-layers 12 \
--encoder-attention-heads 16 \
--encoder-ffn-embed-dim 4096 \
--dropout 0.1 \
--attention-dropout 0.1 \
--act-dropout 0.0
