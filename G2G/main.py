import logging
import argparse
import torch
import datetime
import random
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
import pytorch_warmup as warmup

from utils import set_logging_config, CE_Label_Smooth_Loss, save_checkpoint, \
    de_train_test_split_3fold
from dataloader import *
from model import EncoderNet

# # t-SNE Visualisation
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.manifold import TSNE
# from time import time
# from sklearn.decomposition import PCA

from config import load_config


class Trainer(object):

    def __init__(self, args, dataset_loader):

        self.args = args
        self.dataset_loader = dataset_loader
        self.config = self.args.config

        self.subjects = self.dataset_loader.subjects
        self.sampleList = self.dataset_loader.sampleList
        self.labelList = self.dataset_loader.labelList
        self.clip1 = self.dataset_loader.clip1
        self.clip2 = self.dataset_loader.clip2

    def train(self):
        logger = logging.getLogger("train")
        print("------------------------------------------------------------------------")
        logger.info(
            "Begin experiment on fold: {}".format(str(self.config["cfold"])))

        acc_dic = {}
        acc_list = []
        # 轮询不同的被试，划分训练和测试集
        for subject_index in range(len(self.subjects)):
            subject = self.subjects[subject_index]
            data_and_label = None

            logger.info("Begin {} experiment {} on: {}".format(self.args.mode, str(subject_index+1), subject))

            if self.args.mode == "de":
                data_and_label = de_train_test_split_3fold(self.sampleList[subject_index], self.labelList[subject_index],
                                                     self.clip1[subject_index], self.clip2[subject_index], self.config)

            train_set = TensorDataset((torch.from_numpy(data_and_label["x_train"])).float(),
                                      (torch.from_numpy(data_and_label["y_train"])).long())
            val_set = TensorDataset((torch.from_numpy(data_and_label["x_test"])).float(),
                                    (torch.from_numpy(data_and_label["y_test"])).long())

            train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

            encoder = EncoderNet(self.args)
            g2g_params, backbone_params, fc_params = [], [], []
            for pname, p in encoder.named_parameters():
                if "relation" in str(pname):
                    g2g_params += [p]
                elif "backbone" in str(pname):
                    backbone_params += [p]
                else:
                    fc_params += [p]

            optimizer = optim.AdamW([
                {'params': g2g_params, 'lr': self.args.lr/1.0},
                {'params': backbone_params, 'lr': self.args.lr/1.0},
                {'params': fc_params, 'lr': self.args.lr/1.0},
            ], betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

            # 使用label smooth
            _loss = CE_Label_Smooth_Loss(classes=self.args.config["num_class"], epsilon=self.args.config["epsilon"]).to(self.args.device)

            encoder = encoder.to(self.args.device)
            # warm up
            num_steps = len(train_loader) * self.args.epochs
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_steps // 3], gamma=0.1)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
            warmup_scheduler.last_step = -1

            best_val_acc = 0
            for epoch in range(self.args.epochs):
                train_acc, train_loss, val_loss, val_acc = 0, 0, 0, 0

                encoder.train()
                for item, (x, y) in enumerate(train_loader):
                    lr_scheduler.step(epoch - 1)
                    warmup_scheduler.dampen()

                    encoder.zero_grad()
                    x, y = x.to(self.args.device), y.to(self.args.device, dtype=torch.int64)
                    output = encoder(x) # 返回最后一层的结果和各层的结果
                    loss = _loss(output, y)
                    loss.backward()
                    optimizer.step()
                    train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == y.cpu().data.numpy())
                    train_loss += loss.item()
                tr_acc = round(float(train_acc / train_set.__len__()), 4)
                tr_loss = round(float(train_loss / train_set.__len__()), 4)

                encoder.eval()
                with torch.no_grad():
                    for j, (a, b) in enumerate(val_loader):
                        a, b = a.to(self.args.device), b.to(self.args.device, dtype=torch.int64)
                        output = encoder(a)
                        batch_loss = _loss(output, b)
                        val_loss += batch_loss.item()
                        val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == b.cpu().data.numpy())

                val_acc = round(float(val_acc / val_set.__len__()), 4)
                val_loss = round(float(val_loss / val_set.__len__()), 4)
                if epoch % 1 == 0:
                    logger.info(f"Epoch: {epoch}, train_acc: {tr_acc}, train_loss: {tr_loss}, val_acc: {val_acc}, val_loss: {val_loss}")

                is_best_acc = 0
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    # whether to update the best model
                    is_best_acc = 1

                # save checkpoints (best and newest)
                save_checkpoint({
                    'iteration': epoch,
                    'enc_module_state_dict': encoder.state_dict(),
                    'test_acc': val_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best_acc, self.args.log_dir)

                if best_val_acc == 1:
                    break

            acc_dic[str(subject)] = best_val_acc
            acc_list.append(best_val_acc)

            logger.info("Current best acc is : {}".format(acc_dic))
            print("g2g_params:" + str(optimizer.state_dict()['param_groups'][0]['lr']),
                  "; backbone_params:" + str(optimizer.state_dict()['param_groups'][1]['lr']),
                  "; fc_params:" + str(optimizer.state_dict()['param_groups'][2]['lr']))
            logger.info(self.args)
            logger.info("Current average acc is : {}, std is : {}".format(np.mean(acc_list), np.std(acc_list, ddof = 1)))


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:7", help="gpu device")
    parser.add_argument("--config", type=str, default=os.path.join(".", "config.py"), help="")
    parser.add_argument("--log_dir", type=str, default=os.path.join(".", "logs",), help="log file dir")
    parser.add_argument("--seed", type=int, default=222, help="random seed")
    parser.add_argument("--dataset", type=str, default="SEED5", help="dataset: SEED, seed4, SEED5, MPED, bcic ")
    parser.add_argument("--mode", type=str, default="de", help="dependent(de) or independent(inde)")
    parser.add_argument("--best_step", type=int, default=0, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for supervised learning")
    parser.add_argument("--lr", type=float, default=0.008, help="")
    parser.add_argument("--head_num", type=int, default=6, help="head num")
    parser.add_argument("--epochs", type=int, default=300, help="")
    parser.add_argument("--rand_ali_num", type=int, default=2, help="random aligment num") # 需要同时使用几个随机排列
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--backbone", type=str, default="ResNet18", help="selected cnn backbone")

    args = parser.parse_args()

    #load config file
    args.config = load_config(args.dataset)

    datetime_path = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d-%H:%M:%S'))
    args.log_dir = os.path.join(args.log_dir, args.dataset, datetime_path)

    set_logging_config(args.log_dir)
    logger = logging.getLogger("main")

    logger.info("Current dataset config：" + str(args.dataset))
    logger.info("Launching experiment on: {} {}pendent mode".format(args.dataset, args.mode))
    logger.info("Generated logs and checkpoints will be saved to：{}".format(args.log_dir))
    # logger.info("Generated checkpoints will be saved to:{}".format(os.path.join(args.checkpoint_dir, args.dataset)))
    print()

    logger.info("---------------conmand line arguments and configs----------------")
    logger.info(args)
    print("Is cuda available : {}".format(torch.cuda.is_available()))

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # 由于环境不匹配后补的代码

    dataset_loader = None # 预先声明一下实验数据
    if args.dataset == "SEED":
        dataset_loader = SEED(args.config)
    elif args.dataset == "SEED5":
        dataset_loader = SEED5(args.config)
    elif args.dataset == "MPED":
        dataset_loader = MPED(args.config)
    else:
        args.config["dataset_name"] = "NOT SELECTED"

    # add checkpoint
    if not os.path.join(args.log_dir, "model_best.pth.tar"):
        args.best_step = 0
    else:
        pass

    # define trainer
    trainer = Trainer(args = args,
                      dataset_loader = dataset_loader)
    trainer.train()


if __name__ == "__main__":
    main()

