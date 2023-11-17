import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="tmp_logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-L/14') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+id+mlm', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm','mse']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(224, 224)) ## COCO
    # parser.add_argument("--img_size", type=tuple, default=(384, 128)) ## RSTPReid
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    # parser.add_argument("--vocab_size", type=int, default=49408)
    parser.add_argument("--vocab_size", type=int, default=26964)


    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="RSTPReid", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    ######################## tokengt ########################
    """Add model-specific arguments to the parser."""
    parser.add_argument("--dropout", type=float, default=0.0, metavar="D",  help="dropout prob")
    parser.add_argument("--attention-dropout", type=float, default=0.1, metavar="D", help="dropout prob for attention weights")
    parser.add_argument("--act-dropout", type=float, default=0.1, metavar="D", help="dropout prob after activation in FFN")

    parser.add_argument("--encoder-ffn-embed-dim", type=int, default=768,  metavar="N", help="encoder embedding dim for FFN")
    parser.add_argument("--encoder-layers", type=int, default=12, metavar="N", help="num encoder layers")
    parser.add_argument("--encoder-attention-heads", type=int, default=32, metavar="N", help="num encoder attention heads")
    parser.add_argument("--encoder-embed-dim", type=int, default=768, metavar="N", help="encoder embedding dimension")
    parser.add_argument("--share-encoder-input-output-embed", default=False, action="store_true",
                        help="share encoder input and output embeddings")

    parser.add_argument("--rand-node-id", default=False,action="store_true", help="use random feature node identifiers")
    parser.add_argument("--rand-node-id-dim", default=64, type=int, metavar="N", help="dim of random node identifiers")
    parser.add_argument("--orf-node-id", default=False, action="store_true", help="use orthogonal random feature node identifiers")
    parser.add_argument("--orf-node-id-dim", default=64, type=int, metavar="N", help="dim of orthogonal random node identifier")
    parser.add_argument("--lap-node-id", default=True, action="store_true", help="use Laplacian eigenvector node identifiers")
    parser.add_argument("--lap-node-id-k", default=16, type=int, metavar="N",
                        help="number of Laplacian eigenvectors to use, from smallest eigenvalues")
    parser.add_argument("--lap-node-id-sign-flip", default=True, action="store_true", help="randomly flip the signs of eigvecs")
    parser.add_argument("--lap-node-id-eig-dropout", default=0.2, type=float, metavar="D", help="dropout prob for Lap eigvecs")
    parser.add_argument("--type-id", default=False, action="store_true", help="use type identifiers")

    parser.add_argument("--stochastic-depth", default=False, action="store_true", help="use stochastic depth regularizer")

    parser.add_argument("--performer", action="store_true", help="linearized self-attention with Performer kernel")
    parser.add_argument("--performer-nb-features", type=int, metavar="N",
                        help="number of random features for Performer, defaults to (d*log(d)) where d is head dim")
    parser.add_argument("--performer-feature-redraw-interval", type=int, metavar="N",
                        help="how frequently to redraw the projection matrix for Performer")
    parser.add_argument("--performer-generalized-attention", action="store_true",
                        help="defaults to softmax approximation, but can be set to True for generalized attention")
    parser.add_argument("--performer-finetune", action="store_true",
                        help="load softmax checkpoint and fine-tune with performer")

    parser.add_argument("--apply-graphormer-init", default=False, action="store_true", help="use Graphormer initialization")
    parser.add_argument("--activation-fn", default="gelu", help="activation to use")
    parser.add_argument("--encoder-normalize-before", default=True, action="store_true", help="apply layernorm before encoder")
    parser.add_argument("--prenorm", default=True, action="store_true", help="apply layernorm before self-attention and ffn")
    parser.add_argument("--postnorm", default=False, action="store_true", help="apply layernorm after self-attention and ffn")
    parser.add_argument("--return-attention", default=False, action="store_true", help="obtain attention maps from all layers",)

    
    ######################## MAE ###########################
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--patch_size', default=4, type=int)
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    # parser.add_argument('--input_size', default=224, type=int,
    #                     help='images input size for backbone')
    # parser.add_argument('--input_size', default=(384, 128), type=tuple,
    #                     help='images input size for backbone')
    parser.add_argument('--input_size', default=(364, 364), type=tuple,
                        help='images input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
    #                     help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    # parser.add_argument('--weight_decay', type=float, default=0.05,
    #                     help='weight decay (default: 0.05)')
    # parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    #     weight decay. We use a cosine schedule for WD. 
    #     (Set the same value with args.weight_decay to keep weight decay no change)""")

    # parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
    #                     help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/data2/IRRA/COCO/images/', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    # parser.add_argument('--output_dir', default='',
    #                     help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    

    args = parser.parse_args()

    return args