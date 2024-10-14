import argparse
from params import *
from dataloader import *
from mtl_method import AdvMSMTL
import torch
import random
import numpy as np

def parser_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--dataset', default='Campus_multi-domain_rtype', type=str,
                        help='choose the dataset for training, pacs, office_caltech or office_home')

    
    parser.add_argument('--train_bs', default=32, type=int, help='batch size for training')  # modify
    parser.add_argument('--valid_bs', default=32, type=int, help='batch size for validation')  # modify
    parser.add_argument('--test_bs', default=32, type=int, help='batch size for test') # modify
    
    parser.add_argument('--initial_lr', default=1e-3, type=float)  # initial lr

    parser.add_argument('--ratio', default=0.1, type=float)  # experiments ratio

    parser.add_argument('--down_period', default=5, type=int)
    parser.add_argument('--lr_decay', default=0.95, type=float)

    parser.add_argument('--temp', default=0.001, help= "temperature for con obj", type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    parser.add_argument('--total_round', help = "how many rounds to repeat",default=1, type=int)
    parser.add_argument('--epoch_per_round', default=2, type=int)

    parser.add_argument("--drift_ratio", type=float, help="used for label shift, keep as 0 for no-shift setting", default=0)
    parser.add_argument('--re_weighting', default=True, type=bool)

    parser.add_argument("--seed", default=3407, type=int)

    parser.add_argument("--mtr_margin", type = float, help="mtr_margin", default=1.0)
    parser.add_argument("--mtr_scale_pos", type = float, help="mtr_loss_pos_scale", default=2.0)
    parser.add_argument("--mtr_scale_neg", type = float, help="mtr_loss_neg_loss", default=40.0)
    parser.add_argument("--self_adjust", type = bool, help="auto trade off", default=True)
    parser.add_argument("--apply_meta_decay", type = bool, help="decay meta learning weightt", default=False)


    parser.add_argument("--weight_mtr_loss", type = float, help="weight_mtr_loss", default=1e-5)
    parser.add_argument("--weight_dis_loss", type = float, help="weight_dis_loss", default=0.1)

    parser.add_argument("--gp_param", type = float, help="weight of gradient penalty", default=10.0)
    parser.add_argument("--w_d_round", type = int, help="rounds for dis training", default=1)
    parser.add_argument("--add_mtr", type = int, help = "rounds to start metric learning", default = 5)
    parser.add_argument("--checkpoint_path", type = str, help = "checkpoint path", default = '/home/yourname/mtl/f_code_WA/test_results/checkpoints/best_model.pth')
    parser.add_argument("--start_epoch", type = int, help = "start epoch", default = 0)

    parser.add_argument("--task_index_known", type = bool, help="whether the task index is agnostic to the model", default=False)

    return parser.parse_args()


print("=================Training started==================")

args = parser_params()

print('args for the experiments', args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = torch.cuda.is_available()

def set_seed(seed):
    if seed is not None:
        random.seed(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed) 
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def get_loaders(args):
    config[args.dataset]['batch_size']= args.train_bs
    config[args.dataset]['valid_batch_size']= args.valid_bs
    config[args.dataset]['test_batch_size']= args.test_bs

    if args.dataset == 'RSCD_multi-domain_rtype':
        train_loader, val_loader, test_loader= get_dloader_RSCD(data_name='RSCD_multi-domain_rtype', 
                                                                re_weighting=True, ratio=args.ratio,
                                                                configure=config)
    elif args.dataset == 'Campus_multi-domain_rtype':
        train_loader, val_loader, test_loader= get_dloader_RSCD(data_name='Campus_multi-domain_rtype', 
                                                                re_weighting=True, ratio=args.ratio,
                                                                configure=config)
    return train_loader, val_loader, test_loader



batch_size = config[args.dataset]['batch_size']
args_train = config[args.dataset]

print('GPU: {}'.format(args.gpu))


args_train['dataset'] = args.dataset
args_train['lr'] = args.initial_lr
args_train['down_period'] = args.down_period
args_train['weight_decay'] = args.weight_decay
args_train['temp'] = args.temp
args_train['lr_decay'] = args.lr_decay
task_list = args_train['task_list'].copy()

args_train['param_metric'] = {'scale_pos':args.mtr_scale_pos,
                            'scale_neg':args.mtr_scale_neg,
                            'margin':args.mtr_margin}

args_train['epoch_per_round'] = args.epoch_per_round

args_train['gp_param'] = args.gp_param
args_train['w_d_round'] = args.w_d_round
args_train['weight_metric_loss'] = args.weight_mtr_loss
args_train['weight_dis_loss'] = args.weight_dis_loss

args_train['add_mtr'] = args.add_mtr
args_train['total_epoch'] = args.epoch_per_round

args_train['tsk_idx_known'] = args.task_index_known


task_list.append('Avg'),
task_list.append('loss_ms')
task_list.append('loss_w')
task_list.append('Total_loss')


results_table = {}

train_loader, val_loader, test_loader = get_loaders(args=args)
set_seed(seed=args.seed)

for rd in range(args.total_round):
    MTL_algo = AdvMSMTL(train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        args=args_train,
                        dataset=args.dataset)
    

    best_metrics = {
    'acc': {'avg': 0, 'raw': []},
    'precision': {'avg': 0, 'raw': []},
    'recall': {'avg': 0, 'raw': []},
    'f1': {'avg': 0, 'raw': []}
}
    
    acc_list = []
    epoch_results = {}

    val_acc_list = [] 
    for epoch in range(args.start_epoch, args.start_epoch + args.epoch_per_round):
        print('epoch:', epoch)
        
        result = MTL_algo.model_fit(epoch=epoch)
        loss = result['loss']
        ms_loss = result['loss_ms']
        dis_loss = result['loss_dis']

 
        val_acc, val_precision, val_recall, val_f1 = MTL_algo.model_valid(checkpoint_path=args.checkpoint_path, start_epoch=args.start_epoch, epoch=epoch)
        
        val_avg_acc = np.mean(val_acc)
        val_avg_precision = np.mean(val_precision)
        val_avg_recall = np.mean(val_recall)
        val_avg_f1 = np.mean(val_f1)
        
        val_acc_list.append(val_avg_acc)  
        
        print(f'Validation results - Average accuracy: {val_avg_acc:.4f}, Average precision: {val_avg_precision:.4f}, Average recall: {val_avg_recall:.4f}, Average F1 score: {val_avg_f1:.4f}')
        
        if epoch == args.start_epoch:
            val_best_acc = val_avg_acc
        elif val_avg_acc > val_best_acc:
            val_best_acc = val_avg_acc
            
            torch.save({
                'epoch': epoch,
                'featurizer_state_dict': MTL_algo.featurizer.state_dict(),
                'classifier_state_dict': MTL_algo.classifier.state_dict(),
                'best_val_acc': val_best_acc
            }, args.checkpoint_path)
            
            print(f'Saved new best model, validation accuracy: {val_best_acc:.4f}')
        tasks_teAcc, precision_scores, recall_scores, f1_scores = MTL_algo.model_test(checkpoint_path=args.checkpoint_path, epoch=epoch)

        avg_acc = np.mean(tasks_teAcc)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)

        acc_list.append(avg_acc)
        if avg_acc > best_metrics['acc']['avg']:
           best_metrics['acc'] = {
        'raw': tasks_teAcc.tolist(), 
               'avg': avg_acc
           }
           best_metrics['precision'] = {
               'raw': precision_scores.tolist(),
               'avg': avg_precision
           }
           best_metrics['recall'] = {
               'raw': recall_scores.tolist(),
               'avg': avg_recall
           }
           best_metrics['f1'] = {
               'raw': f1_scores.tolist(), 
               'avg': avg_f1
           }
        
        

