from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import argparse
import os
import random
from torch import optim

from data_DB_L5 import *
from util import *

# model name
from rootmodel.B_EDVR_DB_L5 import *
model = EDVR()

parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
# wandb and project
parser.add_argument("--use_wandb", default=True, action="store_true")
parser.add_argument("--Project_name", default="SPVSR_V5", type=str) # wandb Project name
parser.add_argument("--This_name", default="B_EDVR_DB_L5", type=str) # wandb run name & model save name path
parser.add_argument("--wandb_username", default="tangle", type=str)
# dataset 文件夹要以/结尾
parser.add_argument("--train_root_path", default='datasets/deblur/train/', type=str, help="train root path")
parser.add_argument("--test_root_path", default='datasets/deblur/test/', type=str, help="test root path")
# train setting
parser.add_argument("--cuda", default=True, action="store_true")
parser.add_argument("--cuda_id", default=4, type=int)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--max_epoch", default=10000, type=int)
parser.add_argument("--batchSize", default=16, type=int)
parser.add_argument("--lr", default=0.0004, type=float)
parser.add_argument("--threads", default=8, type=int)
# other setting
parser.add_argument("--dataset_pin_memory", default=True, action="store_true")
parser.add_argument("--dataset_drop_last", default=True, action="store_true")
parser.add_argument("--dataset_shuffle", default=True, action="store_true")
parser.add_argument("--test_save_epoch", default=1, type=int)
parser.add_argument("--decay_loss_epoch", default=10, type=int)
parser.add_argument("--decay_loss_ratio", default=0.8, type=float)
opt = parser.parse_args()



def main():
    global model, opt
    if opt.use_wandb:
        wandb.init(project=opt.Project_name, name=opt.This_name, entity=opt.wandb_username)
    print(opt)

    print("===> Find Cuda")
    cuda = opt.cuda
    if cuda:
        torch.cuda.set_device(opt.cuda_id)
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    # 利用显存换取浮点训练加速
    # cudnn.benchmark = True

    print("===> Loading datasets")

    train_set = train_data_set(opt.train_root_path, batchsize=opt.batchSize)
    train_dataloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, num_workers=opt.threads,
                                  drop_last=True)

    test_set_1 = test_data_set(opt.test_root_path, "000/")
    test_loader_1 = DataLoader(dataset=test_set_1, batch_size=1, num_workers=0)
    test_set_2 = test_data_set(opt.test_root_path, "011/")
    test_loader_2 = DataLoader(dataset=test_set_2, batch_size=1, num_workers=0)
    test_set_3 = test_data_set(opt.test_root_path, "015/")
    test_loader_3 = DataLoader(dataset=test_set_3, batch_size=1, num_workers=0)
    test_set_4 = test_data_set(opt.test_root_path, "020/")
    test_loader_4 = DataLoader(dataset=test_set_4, batch_size=1, num_workers=0)

    test_loader_sum = []
    test_loader_sum.append(test_loader_1)
    test_loader_sum.append(test_loader_2)
    test_loader_sum.append(test_loader_3)
    test_loader_sum.append(test_loader_4)

    # test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)

    print("===> Setting loss")
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    print("===> Do Resume Or Skip")
    # model = get_yu(model, "checkpoints/over/TSALSTM_ATD/model_epoch_212_psnr_27.3702.pth")

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        # 训练
        train(optimizer, model, criterion, epoch, train_dataloader)
        # 测试和保存
        if (epoch+1) % opt.test_save_epoch == 0:
            psnr = test_train_set(model, test_loader_sum, epoch)
            save_checkpoint(model, psnr, epoch, optimizer.param_groups[0]["lr"])
        # 降低学习率
        if (epoch+1) % opt.decay_loss_epoch == 0:
            for p in optimizer.param_groups:
                p['lr'] *= opt.decay_loss_ratio

def train(optimizer, model, criterion, epoch, train_dataloader):
    global opt
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    avg_loss = AverageMeter()
    if opt.cuda:
        model = model.cuda()
    for iteration, batch in enumerate(train_dataloader):
        input, target = batch
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        out = model(input)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())
        if iteration % 50 == 0:
            if opt.use_wandb:
                wandb.log({'epoch': epoch, 'iter_loss': avg_loss.avg})
            print('epoch_iter_{}_loss is {:.10f}'.format(iteration, avg_loss.avg))

def test_train_set(this_model, test_loader_sum, epoch_num):
    print(" -- Start eval --")
    psnr_sum = 0
    ssim_sum = 0
    with torch.no_grad():
        for iii, test_loader in enumerate(test_loader_sum):
            psnr = AverageMeter()
            ssim = AverageMeter()
            model = this_model
            if opt.cuda:
                model = model.cuda()
            model.eval()
            for iteration, batch in enumerate(test_loader, 1):
                input, target = batch
                if opt.cuda:
                    input = input.cuda()
                    target = target.cuda()
                out = model(input)
                psnr.update(calc_psnr(out, target), len(out))
                ssim.update(cal_ssim_tensor(out[0, :, :, :], target[0, :, :, :]), len(out))
            if opt.use_wandb:
                wandb.log({'psnr{}'.format(iii+1): psnr.avg, 'ssim{}'.format(iii+1): ssim.avg})
            print("--->This--{}--epoch:{}--Avg--PSNR: {:.4f} dB -- SSIM :{}--Dir: {}".format(This_name, epoch_num, psnr.avg, ssim.avg, iii+1))
            psnr_sum += psnr.avg
            ssim_sum += ssim.avg
    print(" -- Sum PSNR: {:.4f} -- ".format(psnr_sum/4.))
    if opt.use_wandb:
        wandb.log({'epoch': epoch_num, 'psnr_sum': psnr_sum/4.})
    print(" -- Sum SSIM: {:.4f} -- ".format(ssim_sum / 4.))
    if opt.use_wandb:
        wandb.log({'epoch': epoch_num, 'ssim_sum': ssim_sum / 4.})
    return psnr_sum/4.


def save_checkpoint(model, psnr, epoch, lr):
    global opt

    model_folder = "checkpoints/{}/".format(opt.This_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "epoch_{}_psnr_{:.4f}_lr_{}.pth".format(epoch, psnr, lr)
    torch.save({'model': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()