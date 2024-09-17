""" 
@Date: 2021/07/17
@description:
"""
import os
import torch
import torch.nn as nn
import datetime


class BaseModule(nn.Module):
    def __init__(self, ckpt_dir=None):
        super().__init__()

        self.ckpt_dir = ckpt_dir

        if ckpt_dir:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            else:
                self.model_lst = [x for x in sorted(os.listdir(self.ckpt_dir)) if x.endswith('.pkl')]

        self.last_model_path = None
        self.best_model_path = None
        self.oracle_model_path = None
        self.average_model_path = None
        self.best_accuracy = -float('inf')
        self.average_accuracy = -float('inf')
        self.oracle_accuracy = -float('inf')
        self.acc_d = {}
        self.acc_d_new = {}
        self.acc_d_oracle = {}

    def show_parameter_number(self, logger):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info('{} parameter total:{:,}, trainable:{:,}'.format(self._get_name(), total, trainable))

    def load(self, device, logger, optimizer=None, best=False, option='best'):
        if len(self.model_lst) == 0:
            logger.info('*'*50)
            logger.info("Empty model folder! Using initial weights")
            logger.info('*'*50)
            return 0

        last_model_lst = list(filter(lambda n: '_last_' in n, self.model_lst))
        best_model_lst = list(filter(lambda n: '_best_' in n, self.model_lst))
        oracle_model_lst = list(filter(lambda n: '_oracle_' in n, self.model_lst))
        average_model_lst = list(filter(lambda n: '_average_' in n, self.model_lst))

        # if len(last_model_lst) == 0 and len(best_model_lst) == 0:
        #     logger.info('*'*50)
        #     ckpt_path = os.path.join(self.ckpt_dir, self.model_lst[0])
        #     logger.info(f"Load: {ckpt_path}")
        #     # checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
        #     checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        #     self.load_state_dict(checkpoint, strict=False)
        #     logger.info('*'*50)
        #     return 0

        checkpoint = None
        if len(last_model_lst) > 0:
            self.last_model_path = os.path.join(self.ckpt_dir, last_model_lst[-1])
            # print(self.last_model_path)
            # checkpoint = torch.load(self.last_model_path, map_location=torch.device(device))
            checkpoint = torch.load(self.last_model_path, map_location=torch.device('cpu'))
            self.best_accuracy = checkpoint['accuracy']
            if 'oracle_accuracy' in checkpoint.keys():    
                self.oracle_accuracy = checkpoint['oracle_accuracy']
            if 'average_accuracy' in checkpoint.keys():    
                self.average_accuracy = checkpoint['average_accuracy']
            self.acc_d = checkpoint['acc_d']
            if 'acc_d_new' in checkpoint.keys():
                self.acc_d_new = checkpoint['acc_d_new']
            if 'acc_d_oracle' in checkpoint.keys():
                self.acc_d_oracle = checkpoint['acc_d_oracle']    

        if len(best_model_lst) > 0:
            self.best_model_path = os.path.join(self.ckpt_dir, best_model_lst[-1])
            # best_checkpoint = torch.load(self.best_model_path, map_location=torch.device(device))
            best_checkpoint = torch.load(self.best_model_path, map_location=torch.device('cpu'))
            self.best_accuracy = best_checkpoint['accuracy']
            if 'oracle_accuracy' in best_checkpoint.keys():    
                self.oracle_accuracy = best_checkpoint['oracle_accuracy']
            if 'average_accuracy' in best_checkpoint.keys():    
                self.average_accuracy = best_checkpoint['average_accuracy']
            self.acc_d = best_checkpoint['acc_d']
            if 'acc_d_new' in best_checkpoint.keys():
                self.acc_d_new = best_checkpoint['acc_d_new']
            if 'acc_d_oracle' in best_checkpoint.keys():
                self.acc_d_oracle = best_checkpoint['acc_d_oracle']
            # if best:
            #     checkpoint = best_checkpoint

        if len(oracle_model_lst) > 0:
            self.oracle_model_path = os.path.join(self.ckpt_dir, oracle_model_lst[-1])
            # oracle_checkpoint = torch.load(self.oracle_model_path, map_location=torch.device(device))
            oracle_checkpoint = torch.load(self.oracle_model_path, map_location=torch.device('cpu'))
            self.best_accuracy = oracle_checkpoint['accuracy']
            if 'oracle_accuracy' in oracle_checkpoint.keys():    
                self.oracle_accuracy = oracle_checkpoint['oracle_accuracy']
            if 'average_accuracy' in oracle_checkpoint.keys():    
                self.average_accuracy = oracle_checkpoint['average_accuracy']
            self.acc_d = oracle_checkpoint['acc_d']
            if 'acc_d_new' in oracle_checkpoint.keys():
                self.acc_d_new = oracle_checkpoint['acc_d_new']
            if 'acc_d_oracle' in oracle_checkpoint.keys():
                self.acc_d_oracle = oracle_checkpoint['acc_d_oracle']

        if len(average_model_lst) > 0:
            self.average_model_path = os.path.join(self.ckpt_dir, average_model_lst[-1])
            # average_checkpoint = torch.load(self.average_model_path, map_location=torch.device(device))
            average_checkpoint = torch.load(self.average_model_path, map_location=torch.device('cpu'))
            self.best_accuracy = average_checkpoint['accuracy']
            if 'oracle_accuracy' in average_checkpoint.keys():    
                self.oracle_accuracy = average_checkpoint['oracle_accuracy']
            if 'average_accuracy' in average_checkpoint.keys():    
                self.average_accuracy = average_checkpoint['average_accuracy']
            self.acc_d = average_checkpoint['acc_d']
            if 'acc_d_new' in average_checkpoint.keys():
                self.acc_d_new = average_checkpoint['acc_d_new']
            if 'acc_d_oracle' in average_checkpoint.keys():
                self.acc_d_oracle = average_checkpoint['acc_d_oracle']
        
        if option == 'last':
            checkpoint = checkpoint
        elif option == 'best':
            checkpoint = best_checkpoint
        elif option == 'oracle':
            checkpoint = oracle_checkpoint
        elif option == 'average':
            checkpoint = average_checkpoint    
        else:
            logger.error("Invalid checkpoint option!!!!")
        
        # origin label IOU
        for k in self.acc_d:
            if isinstance(self.acc_d[k], float):
                self.acc_d[k] = {
                    'acc': self.acc_d[k],
                    'epoch': checkpoint['epoch']
                }
        # new label IOU
        if self.acc_d_new is None:
            self.acc_d_new = {}
        if self.acc_d_new is not {}:
            for k in self.acc_d_new:
                if isinstance(self.acc_d_new[k], float):
                    self.acc_d_new[k] = {
                        'acc': self.acc_d_new[k],
                        'epoch': checkpoint['epoch']
                    }        
        # Oracle IOU
        if self.acc_d_oracle is None:
            self.acc_d_oracle = {}
        if self.acc_d_oracle is not {}:
            for k in self.acc_d_oracle:
                if isinstance(self.acc_d_oracle[k], float):
                    self.acc_d_oracle[k] = {
                        'acc': self.acc_d_oracle[k],
                        'epoch': checkpoint['epoch']
                    } 


        if checkpoint is None:
            logger.error("Invalid checkpoint")
            return

        self.load_state_dict(checkpoint['net'], strict=False)
        # if optimizer and not best:  # best的时候使用新的优化器比如从adam->sgd
        #     logger.info('Load optimizer')
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if torch.is_tensor(v):
        #                 state[k] = v.to(device)

        logger.info('*'*50)
        if option == 'best':
            logger.info(f"Load best: {self.best_model_path}")
        elif option == 'oracle':
            logger.info(f"Load oracle: {self.oracle_model_path}")
        elif option == 'average':
            logger.info(f"Load average: {self.average_model_path}")
        else:
            logger.info(f"Load last: {self.last_model_path}")

        logger.info(f"Best accuracy: {self.best_accuracy}")
        logger.info(f"Average accuracy: {self.average_accuracy}")
        logger.info(f"Oracle accuracy: {self.oracle_accuracy}")
        logger.info(f"Last epoch: {checkpoint['epoch'] + 1}")
        logger.info('*'*50)
        return checkpoint['epoch'] + 1

    def update_acc(self, acc_d, epoch, logger):
        logger.info("-" * 100)
        for k in acc_d:
            if k not in self.acc_d.keys() or acc_d[k] > self.acc_d[k]['acc']:
                self.acc_d[k] = {
                    'acc': acc_d[k],
                    'epoch': epoch
                }
            logger.info(f"Update ACC: {k} {self.acc_d[k]['acc']:.4f}({self.acc_d[k]['epoch']}-{epoch})")
        logger.info("-" * 100)
    def update_acc_new(self, acc_d_new, epoch, logger):
        logger.info("-" * 100)
        for k in acc_d_new:
            if k not in self.acc_d_new.keys() or acc_d_new[k] > self.acc_d_new[k]['acc']:
                self.acc_d_new[k] = {
                    'acc': acc_d_new[k],
                    'epoch': epoch
                }
            logger.info(f"Update ACC: {k} {self.acc_d_new[k]['acc']:.4f}({self.acc_d_new[k]['epoch']}-{epoch})")
        logger.info("-" * 100)
    def update_acc_oracle(self, acc_d_oracle, epoch, logger):
        logger.info("-" * 100)
        for k in acc_d_oracle:
            if k not in self.acc_d_oracle.keys() or acc_d_oracle[k] > self.acc_d_oracle[k]['acc']:
                self.acc_d_oracle[k] = {
                    'acc': acc_d_oracle[k],
                    'epoch': epoch
                }
            logger.info(f"Update ACC: {k} {self.acc_d_oracle[k]['acc']:.4f}({self.acc_d_oracle[k]['epoch']}-{epoch})")
        logger.info("-" * 100)        

    def save(self, optim, epoch, accuracy, logger, replace=True, acc_d=None, acc_d_new=None, acc_d_oracle=None, config=None):
        """

        :param config:
        :param optim:
        :param epoch:
        :param accuracy:
        :param logger:
        :param replace:
        :param acc_d: 其他评估数据，visible_2/3d, full_2/3d, rmse...
        :param acc_d_new: 其他评估数据，visible_2/3d, full_2/3d, rmse...
        :param acc_d_oracle: 其他评估数据，visible_2/3d, full_2/3d, rmse...
        :return:
        """
        if acc_d:
            self.update_acc(acc_d, epoch, logger)
        if acc_d_new:
            self.update_acc_new(acc_d_new, epoch, logger) 
        if acc_d_oracle:
            self.update_acc_oracle(acc_d_oracle, epoch, logger)   
        name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_last_{:.4f}_{}'.format(accuracy, epoch))
        name = f"model_{name}.pkl"
        
        # for oracle checkpoint score
        oracle_accuracy = 0.0
        if acc_d_oracle:
            oracle_accuracy = acc_d_oracle['oracle_full_3d'] 
            name_oracle = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_oracle_{:.4f}_{}'.format(oracle_accuracy, epoch))
            name_oracle = f"model_{name_oracle}.pkl"
            
        # for average checkpoint score
        average_accuracy = None
        if acc_d_new:
            average_accuracy = (acc_d['full_3d'] + acc_d_new['new_full_3d'])/2 
            name_average = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_average_{:.4f}_{}'.format(average_accuracy, epoch))
            name_average = f"model_{name_average}.pkl"
            

        checkpoint = {
            'net': self.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy,
            'oracle_accuracy': oracle_accuracy,
            'average_accuracy': average_accuracy,
            'acc_d': acc_d,
            'acc_d_new': acc_d_new,
            'acc_d_oracle': acc_d_oracle,
        }
        # FIXME:: delete always true
        if (True or config.MODEL.SAVE_LAST) and epoch % config.TRAIN.SAVE_FREQ == 0:
            if replace and self.last_model_path and os.path.exists(self.last_model_path):
                os.remove(self.last_model_path)
            self.last_model_path = os.path.join(self.ckpt_dir, name)
            torch.save(checkpoint, self.last_model_path)
            logger.info(f"Saved last model: {self.last_model_path}")

        # for best iou checkpoint ======================================
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            # FIXME:: delete always true
            if True or config.MODEL.SAVE_BEST:
                if replace and self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                self.best_model_path = os.path.join(self.ckpt_dir, name.replace('last', 'best'))
                torch.save(checkpoint, self.best_model_path)
                logger.info("#" * 100)
                logger.info(f"Saved best model: {self.best_model_path}")
                logger.info("#" * 100)
        
        # for oracle accuracy checkpoint ======================================
        if acc_d_oracle:
            if acc_d_oracle['oracle_full_3d'] > self.oracle_accuracy:
                self.oracle_accuracy = acc_d_oracle['oracle_full_3d']
                # FIXME:: delete always true
                if True or config.MODEL.SAVE_BEST:
                    if replace and self.oracle_model_path and os.path.exists(self.oracle_model_path):
                        os.remove(self.oracle_model_path)
                    self.oracle_model_path = os.path.join(self.ckpt_dir, name_oracle)
                    torch.save(checkpoint, self.oracle_model_path)
                    logger.info("#" * 100)
                    logger.info(f"Saved oracle model: {self.oracle_model_path}")
                    logger.info("#" * 100)    

        # for average accuracy checkpoint ======================================
        if average_accuracy is not None:
            if average_accuracy > self.average_accuracy:
                self.average_accuracy = average_accuracy
                # FIXME:: delete always true
                if True or config.MODEL.SAVE_BEST:
                    if replace and self.average_model_path and os.path.exists(self.average_model_path):
                        os.remove(self.average_model_path)
                    self.average_model_path = os.path.join(self.ckpt_dir, name_average)
                    torch.save(checkpoint, self.average_model_path)
                    logger.info("#" * 100)
                    logger.info(f"Saved average model: {self.average_model_path}")
                    logger.info("#" * 100)               