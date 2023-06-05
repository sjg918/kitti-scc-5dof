
import random
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.maincfg import cfg
from src.fen import FeatureExtractionNetwork_dualpath
from src.can import CrossAttentionNet_base

from src.kittifactory import DataFactory
from src.utils import *

# nohup python trainkitti.py 1> /dev/null 2>&1 &

def train():
    # start
    print(cfg.model)

    # cross attention volume
    back = FeatureExtractionNetwork_dualpath().to(cfg.devices[0])
    Anet = CrossAttentionNet_base(cfg.volume_in_channels, cfg.qkv_channels).to(cfg.devices[0])

    back = nn.DataParallel(back, device_ids=cfg.devices)
    Anet = nn.DataParallel(Anet, device_ids=cfg.devices)

    # define dataloader
    kitti_dataset = DataFactory(cfg)
    kitti_loader = DataLoader(kitti_dataset, batch_size=cfg.batchsize, shuffle=True, num_workers=cfg.num_cpu,\
       pin_memory=True, drop_last=True, worker_init_fn=seed_worker, collate_fn=kitti_dataset.collate_fn_cpu)

    # define optimizer and scheduler
    back_optimizer = optim.Adam(back.parameters(), lr=cfg.learing_rate, betas=(0.9, 0.999), eps=1e-08)
    back_scheduler = optim.lr_scheduler.MultiStepLR(back_optimizer, milestones=cfg.MultiStepLR_milstone, gamma=cfg.MultiStepLR_gamma)
    Anet_optimizer = optim.Adam(Anet.parameters(), lr=cfg.learing_rate, betas=(0.9, 0.999), eps=1e-08)
    Anet_scheduler = optim.lr_scheduler.MultiStepLR(Anet_optimizer, milestones=cfg.MultiStepLR_milstone, gamma=cfg.MultiStepLR_gamma)

    # define loss function
    # model contains loss

    back.train()
    Anet.train()

    left_runningloss_all = []
    right_runningloss_all = []

    for epoch in range(1, cfg.maxepoch+1):
        # print milestones
        print('({} / {}) epoch\n'.format(epoch, cfg.maxepoch))
        with open(cfg.logdir + 'log.txt', 'a') as writer:
            writer.write('({} / {}) epoch\n'.format(epoch, cfg.maxepoch))

        for cnt, train_input_dict in enumerate(kitti_loader):
            left_input = train_input_dict['left'].to(device=cfg.devices[0])
            right_input = train_input_dict['right'].to(device=cfg.devices[0])
            left_targetQuaternion = train_input_dict['left_tgt_Q'].to(device=cfg.devices[0])
            right_targetQuaternion = train_input_dict['right_tgt_Q'].to(device=cfg.devices[0])
            
            # forward
            left_fea = back(left_input)
            right_fea = back(right_input)
            left_loss, right_loss = Anet(left_fea, right_fea, left_targetQuaternion, right_targetQuaternion)
        
            left_loss = left_loss.mean()
            right_loss = right_loss.mean()
            loss = left_loss + right_loss
            # backward
            loss.backward()
            back_optimizer.step()
            back.zero_grad()
            Anet_optimizer.step()
            Anet.zero_grad()
            
            left_runningloss_all.append(left_loss.item())      
            right_runningloss_all.append(right_loss.item())      
            # print steploss
            print("{}/{} losses:left {:.5f} /right {:.5f} ".format(
                cnt, len(kitti_dataset) / cfg.batchsize,
                sum(left_runningloss_all) / len(left_runningloss_all),
                sum(right_runningloss_all) / len(right_runningloss_all),
                ),end="\r")
            continue

        # learning rate scheduling
        back_scheduler.step()
        Anet_scheduler.step()
        
        print("{}/{} losses:left {:.5f} /right {:.5f} ".format(
                cnt, len(kitti_dataset) / cfg.batchsize,
                sum(left_runningloss_all) / len(left_runningloss_all),
                sum(right_runningloss_all) / len(right_runningloss_all),
                ))
        with open(cfg.logdir + 'log.txt', 'a') as writer:
            writer.write("{}/{} losses:left {:.5f} /right {:.5f} \n".format(
                cnt, len(kitti_dataset) / cfg.batchsize,
                sum(left_runningloss_all) / len(left_runningloss_all),
                sum(right_runningloss_all) / len(right_runningloss_all),
                ))

        left_runningloss_all = []
        right_runningloss_all = []

        # save model
        if epoch % 5 == 0:
            torch.save(back.module.state_dict(), cfg.logdir + 'back_' + str(epoch) + '.pth')
            torch.save(Anet.module.state_dict(), cfg.logdir + 'Anet_' + str(epoch) + '.pth')
            print('{} epoch model saved !\n'.format(epoch))
            with open(cfg.logdir + 'log.txt', 'a') as writer:
                writer.write('{} epoch model saved !\n'.format(epoch))
        continue
    # end.


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(777)
    np.random.seed(777)

    if os.path.exists(cfg.logdir):
       pass
    else:
        os.makedirs(cfg.logdir)

    with open(cfg.logdir + 'log.txt', 'w') as writer:
       writer.write("-start-\t")
       writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
       writer.write('\n\n')

    print("\n-start- ", "(", datetime.datetime.now(), ")")
    
    torch.multiprocessing.set_start_method('spawn')
    train()

    print("\n-end- ", "(", datetime.datetime.now(), ")")

    with open(cfg.logdir + 'log.txt', 'a') as writer:
        writer.write('-end-\t')
        writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

