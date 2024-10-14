import os
import time
from datetime import datetime
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from util import *
from networks import *



class AdvMSMTL():

    def __init__(self, train_loader, val_loader, test_loader, args, dataset):
        self.args = args
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_tsk = len(self.args['task_list'])
        self.n_class = self.args['num_classes']


        self.gp_param = self.args['gp_param'] # 10

        self.multi_similarity_loss = MultiSimilarityLoss(self.args['param_metric'])


        self.weight_decay = self.args['weight_decay']
        self.down_period = self.args['down_period']


        self.batch_size = self.args['batch_size']

        self.w_d_round = self.args['w_d_round']
        self.weight_dis_loss = self.args['weight_dis_loss']
        self.gp_param = self.args['gp_param']
        
        
        self.weight_metric_loss = self.args['weight_metric_loss']
        self.add_mtr = self.args['add_mtr']

        self.lr_decay = self.args['lr_decay']

        self.total_epoch = self.args['total_epoch']

        self.tsk_idx_known = self.args['tsk_idx_known']
        print("task index known or not: ", self.tsk_idx_known)

        self.featurizer = AlexNetFc_modified().to(self.device)
 
        self.classifier = CLS(in_dim=512, out_dim=self.n_class).to(self.device) 
   
        self.discriminator = Alexnet_Discriminator_network(512).to(self.device)  
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.fea_lr = self.args['lr']
        self.cls_lr = self.args['lr']
        self.dis_lr = self.args['lr']

        self.vis_interval = 50
        self.domain_label = {0: 'noon', 1:'dusk', 2: 'night'}


    def model_fit(self, epoch):

        runing_mtr_loss = []

        if (epoch+1)%self.down_period ==0:
            self.fea_lr = self.fea_lr*self.lr_decay
            self.dis_lr = self.dis_lr*self.lr_decay
            self.cls_lr = self.cls_lr*self.lr_decay
          
        self.opt_fea = optim.Adam(self.featurizer.parameters() , lr=self.fea_lr)
        self.opt_clf = optim.Adam(self.classifier.parameters(), lr=self.cls_lr,weight_decay=self.weight_decay)
        self.opt_dis = optim.Adam(self.discriminator.parameters(), lr=self.dis_lr)

        count = 0
        for minibatches in zip(*self.train_loader):

            self.featurizer.train()
            self.classifier.train()
            self.discriminator.train()

            num_mb = len(minibatches) 

            input_list = [batch[0] for batch in minibatches]
            target_list = [batch[1] for batch in minibatches]


            if self.tsk_idx_known == True: 
                loss = torch.tensor([0.0]).to(self.device)
                for tn in range(self.num_tsk): 
                    inputs = minibatches[tn][0].to(self.device)
                    targets = minibatches[tn][1].to(self.device)

                    num_batches = get_num_batches(num_mb, minibatches)

                    p = float(count + (epoch+1)* num_batches) / (self.total_epoch )/ num_batches
                    trade_off = 2. / (1. + np.exp(-10 * p)) - 1

                    source_fea = self.featurizer(inputs)[0]   
                    _, fc1_s, fc2_s, predict_prob_source = self.classifier(source_fea)
                    ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, targets)
                    cls_loss = torch.mean(ce, dim=0, keepdim=True)           
                    fea = self.featurizer(inputs)[1]
                    dis_loss = compute_wasserstein(fea, btch_sz = self.batch_size, feature_extractor = self.featurizer, 
                                                   discriminator=self.discriminator ,use_gp= True, gp_weight= self.gp_param)
                    dis_loss = torch.tensor(dis_loss).to(self.device)
                    
                    if epoch > self.add_mtr:
                                          
                        mtr_out = F.normalize(fc1_s, p=2, dim=1)

                        mtr_loss = self.multi_similarity_loss(mtr_out,targets)

                        runing_mtr_loss.append(mtr_loss.item())
                        add_loss = cls_loss + self. weight_dis_loss * trade_off * dis_loss + self.weight_metric_loss* mtr_loss 
                    else:
                        add_loss = cls_loss + self. weight_dis_loss * trade_off * dis_loss # + self.weight_metric_loss* mtr_loss 
                        runing_mtr_loss.append(0)
                    
                    loss += add_loss
                
                loss = loss / self.num_tsk
                
                
            if self.tsk_idx_known == False:
                if count == 0:
                    print("Shuffle the order of task indices")
                input_list, target_list = shuffle_minibatch(input_list, target_list)
                inputs = torch.cat(input_list).to(self.device)
                targets = torch.cat(target_list).to(self.device)
    

                # for t-SNE visualization
                if epoch % self.vis_interval == 0:

                    vis_inputs = inputs.clone().detach().cpu().numpy()
                    num_samples = vis_inputs.shape[0]
                    
                    num_samples_per_task = num_samples // num_mb

                    task_label = np.concatenate([np.full((num_samples_per_task, 1), i) for i in range(num_mb)])
                    


                num_batches = get_num_batches(num_mb, minibatches)

                p = float(count + (epoch+1)* num_batches) / (self.total_epoch )/ num_batches
                trade_off = 2. / (1. + np.exp(-10 * p)) - 1

                source_fea = self.featurizer(inputs)[0]  


                # for t-SNE visualization
                if epoch == 0 and count == 0:
                    vis_fea = source_fea.clone().cpu().detach().numpy()
                    vis_fea = vis_fea.reshape(vis_fea.shape[0], -1)

                    vis_label = task_label

                    scaler = StandardScaler()
                    vis_fea_scaled = scaler.fit_transform(vis_fea)

                    tsne = TSNE(n_components=2, init='pca', learning_rate='auto',
                            random_state=0, perplexity=30, n_iter=1000, method='barnes_hut')

                    details = tsne.fit_transform(vis_fea_scaled)

                    colors = {0: 'green', 1: 'red', 2: 'blue'}
                            
                    plt.figure()

                    unique_label = np.unique(vis_label)
                    for label in unique_label:
                        indice = (vis_label.flatten() == label)
                        plt.scatter(details[indice, 0], details[indice, 1], color=colors[label], label=self.domain_label[label], s=7)

                    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))

                    plt.axis('off')

                    now = datetime.now()
                    dt_string = now.strftime("%Y-%m-%d_%H_%M")

                    save_path = os.path.join('/home/cyy/cyy/mtl/f_code_WA/test_results/t-SNE', 'epoch'+str(epoch)+'_initial_'+str(dt_string))
                    
                    plt.savefig(save_path, dpi=600, format='png')


                    
                _, fc1_s, fc2_s, predict_prob_source = self.classifier(source_fea)

                ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, targets)
                cls_loss = torch.mean(ce, dim=0, keepdim=True)           
                fea = self.featurizer(inputs)[1]
                dis_loss = compute_wasserstein(fea, btch_sz = self.batch_size, discriminator=self.discriminator ,use_gp= True, gp_weight= self.gp_param)

                if epoch > self.add_mtr:
                    mtr_out = F.normalize(fc1_s, p=2, dim=1)
                    mtr_loss = self.multi_similarity_loss(mtr_out,targets)
                    runing_mtr_loss.append(mtr_loss.item())
                    loss = cls_loss + self. weight_dis_loss * trade_off * dis_loss + self.weight_metric_loss* mtr_loss 
                else:
                    loss = cls_loss + self. weight_dis_loss * trade_off * dis_loss # + self.weight_metric_loss* mtr_loss 
                    runing_mtr_loss.append(0)

            loss.backward()
            self.opt_fea.step()
            self.opt_clf.step()
            self.opt_dis.step()
            
            count += 1


        
        # t-SNE visualization during training
        if epoch > 0 and epoch % self.vis_interval == 0:
            vis_fea = source_fea.clone().cpu().detach().numpy()
            vis_fea = vis_fea.reshape(vis_fea.shape[0], -1)

            vis_label = task_label

            scaler = StandardScaler()
            vis_fea_scaled = scaler.fit_transform(vis_fea)

            tsne = TSNE(n_components=2, init='pca', learning_rate='auto',
                    random_state=0, perplexity=30, n_iter=1000, method='barnes_hut')

            details = tsne.fit_transform(vis_fea_scaled)
            
            colors = {0: 'green', 1: 'red', 2: 'blue'}
            
            plt.figure()

            unique_label = np.unique(vis_label)
            for label in unique_label:
                    indice = (vis_label.flatten() == label)
                    plt.scatter(details[indice, 0], details[indice, 1], color=colors[label], label=self.domain_label[label],s=7)

            plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))

            plt.axis('off')


            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d_%H_%M")

            save_path = os.path.join('/home/cyy/cyy/mtl/f_code_WA/test_results/t-SNE', 'epoch'+str(epoch)+'_training_'+str(dt_string))
            
            plt.savefig(save_path, dpi=600, format='png')
   
        if epoch>self.add_mtr:
            return {'loss': loss.item(), 'loss_dis': dis_loss.item(), 'loss_ms': mtr_loss.item()}
        else:
            return {'loss': loss.item(), 'loss_dis': dis_loss.item(), 'loss_ms': 0}
        #pass

    def model_valid(self, checkpoint_path, start_epoch, epoch):
        with torch.no_grad():
            data_loader = self.val_loader

            print(f'Validation epoch {epoch}')
            loss_hypo_value = np.zeros(self.num_tsk)
            correct_hypo = np.zeros(self.num_tsk)
            precision_scores = np.zeros(self.num_tsk)
            recall_scores = np.zeros(self.num_tsk)
            f1_scores = np.zeros(self.num_tsk)

            self.featurizer.eval()
            self.classifier.eval()

            for t in range(self.num_tsk):
                task_preds = []
                task_targets = []
                n_batch_t = 0

                for inputs, targets in (data_loader[t]):
                    n_batch_t += 1
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    if epoch == start_epoch:
                        checkpoint = torch.load(checkpoint_path)
                        self.featurizer.load_state_dict(checkpoint['featurizer_state_dict'])
                        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
                    
                    with torch.no_grad():
                        features = self.featurizer(inputs)[0]
                        label_prob = self.classifier(features)[3]
                    
                    pred = label_prob.argmax(dim=1, keepdim=True)
                    correct_hypo[t] += ((pred.eq(targets.view_as(pred)).sum().item()) / len(pred))
                    loss_hypo_value[t] += F.cross_entropy(label_prob, targets, reduction='mean').item()

                    task_preds.extend(pred.cpu().numpy())
                    task_targets.extend(targets.cpu().numpy())
                
                task_preds_flat = np.array([pred.item() for pred in task_preds])
                
                loss_hypo_value[t] /= n_batch_t
                correct_hypo[t] /= n_batch_t

                precision = precision_score(task_targets, task_preds, average='macro', zero_division=1)
                recall = recall_score(task_targets, task_preds, average='macro')
                f1 = f1_score(task_targets, task_preds, average='macro')

                precision_scores[t] = precision
                recall_scores[t] = recall
                f1_scores[t] = f1

            self.featurizer.train()
            self.classifier.train()

            print('\t ======== EPOCH:{}'.format(epoch))
            print('\t === hypothesis ** validation ** loss \n' + str(loss_hypo_value))
            print('\t === hypothesis ** validation ** accuracy \n' + str(correct_hypo * 100))

            return correct_hypo, precision_scores, recall_scores, f1_scores

    def model_test(self, checkpoint_path, epoch, t_vis=False):
        with torch.no_grad():
            data_loader = self.test_loader

            print(f'Test epoch {epoch}')
            loss_hypo_value = np.zeros(self.num_tsk)
            correct_hypo = np.zeros(self.num_tsk)
            precision_scores = np.zeros(self.num_tsk)
            recall_scores = np.zeros(self.num_tsk)
            f1_scores = np.zeros(self.num_tsk)

            self.featurizer.eval()
            self.classifier.eval()

            checkpoint = torch.load(checkpoint_path)
            self.featurizer.load_state_dict(checkpoint['featurizer_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

            vis_fea = []
            vis_label = []

            for t in range(self.num_tsk):
                task_preds = []
                task_targets = []
                n_batch_t = 0

                for inputs, targets in (data_loader[t]):
                    n_batch_t += 1
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    with torch.no_grad():
                        features = self.featurizer(inputs)[0]
                        label_prob = self.classifier(features)[3]             
                    print(f"confidence: {label_prob.max()}")
                    
                    pred = label_prob.argmax(dim=1, keepdim=True)
                    print(f"pred: {pred}, targets: {targets}")

                    time.sleep(0.5)
                    correct_hypo[t] += ((pred.eq(targets.view_as(pred)).sum().item()) / len(pred))
                    loss_hypo_value[t] += F.cross_entropy(label_prob, targets, reduction='mean').item()

                    task_preds.extend(pred.cpu().numpy())
                    task_targets.extend(targets.cpu().numpy())

                    if t_vis and epoch % self.vis_interval == 0:
                        fea = features.clone().cpu().detach().numpy()
                        fea = fea.reshape(fea.shape[0], -1)
                        task_label = np.full((fea.shape[0], 1), t)
                        vis_fea.append(fea)
                        vis_label.append(task_label)
                
                task_preds_flat = np.array([pred.item() for pred in task_preds])
                
                loss_hypo_value[t] /= n_batch_t
                correct_hypo[t] /= n_batch_t

                precision = precision_score(task_targets, task_preds, average='macro', zero_division=1)
                recall = recall_score(task_targets, task_preds, average='macro')
                f1 = f1_score(task_targets, task_preds, average='macro')

                precision_scores[t] = precision
                recall_scores[t] = recall
                f1_scores[t] = f1

            if t_vis and epoch % self.vis_interval == 0:
                self._visualize_tsne(vis_fea, vis_label, epoch)

            self.featurizer.train()
            self.classifier.train()

            print('\t ======== EPOCH:{}'.format(epoch))
            print('\t === hypothesis ** test ** loss \n' + str(loss_hypo_value))
            print('\t === hypothesis ** test ** accuracy \n' + str(correct_hypo * 100))

            return correct_hypo, precision_scores, recall_scores, f1_scores

    def _visualize_tsne(self, vis_fea, vis_label, epoch):
        vis_fea = np.array(vis_fea)
        vis_fea = np.concatenate(vis_fea, axis=0)
        vis_label = np.concatenate(vis_label, axis=0)
        
        scaler = StandardScaler()
        vis_fea_scaled = scaler.fit_transform(vis_fea)

        tsne = TSNE(n_components=2, init='pca', learning_rate='auto',
                random_state=0, perplexity=30, n_iter=1000, method='barnes_hut')

        details = tsne.fit_transform(vis_fea_scaled)
        
        colors = {0: 'green', 1: 'red', 2: 'blue'}
        
        plt.figure()

        unique_label = np.unique(vis_label)
        for label in unique_label:
            indice = (vis_label.flatten() == label)
            plt.scatter(details[indice, 0], details[indice, 1], color=colors[label], label=self.domain_label[label], s=8)

        plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
        plt.axis('off')

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H_%M")

        save_path = os.path.join('/home/cyy/cyy/mtl/f_code_WA/test_results/t-SNE', f'epoch{epoch}_test_{dt_string}')
        
        plt.savefig(save_path, dpi=600, format='png')