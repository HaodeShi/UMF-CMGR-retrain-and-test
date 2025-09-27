
import sys
sys.path.append("..")
import sys 
import os
sys.path.append("..")
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)

import pathlib
import warnings
import argparse
import numpy
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms

from tqdm import tqdm
import torch.nn.functional

from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform
from dataloader.reg_data import RegData
from models.deformable_net import DeformableNet
from loss.reg_losses import LossFunction_Dense


def hyper_args():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Corss-modality Registration")
    # dataset
    parser.add_argument('--ir', default='dataset/Roadscene/ir_256', type=pathlib.Path) # 源红外图像
    parser.add_argument('--it', default='dataset/Roadscene/Wei_hongwai/rgb2ir_paired_Road_edge_pretrained/test_latest/images', type=pathlib.Path) # 伪红外图像
    parser.add_argument("--batchsize", type=int, default=16, help="training batch size")
    parser.add_argument("--nEpochs", type=int, default=800, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")
    parser.add_argument("--step", type=int, default=1200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--pretrained", default="../cache/Reg_only/220506_Deformable_2*Fe_10*Grad/cp_0800.pth", type=str, help="path to pretrained model (default: none)")
    parser.add_argument("--ckpt", default="../cache/Reg_only/220507_Deformable_2*Fe_10*Grad", type=str, help="path to pretrained model (default: none)")
    args = parser.parse_args()
    return args


def main(args):
    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)

    print("===> Loading datasets")
    crop = torchvision.transforms.RandomResizedCrop(256)
    data = RegData(args.ir, args.it, crop)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)

    print("===> Building model")
    net = DeformableNet().cuda()
    criterion = LossFunction_Dense().cuda()

    print("===> Setting Optimizer")
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)

    print("===> Building deformation")
    affine = AffineTransform(translate=0.01)
    elastic = ElasticTransform(kernel_size=101, sigma=16)

    # 加载预训练模型
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading model '{}'".format(args.pretrained))
            model_state_dict = torch.load(args.pretrained)
            net.load_state_dict(model_state_dict)
        else:
            print("=> no model found at '{}'".format(args.pretrained))

    print("===> Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        train(training_data_loader, optimizer, net, criterion, epoch, elastic, affine, args)
        if epoch % 20 == 0:
            save_checkpoint(net, epoch, cache)

def train(training_data_loader, optimizer, net, criterion, epoch, elastic, affine, args):
    net.train()
    tqdm_loader = tqdm(training_data_loader, disable=True)

    # 更新学习率
    lr = adjust_learning_rate(optimizer, epoch - 1, args)
    print("Epoch={}, lr={}".format(epoch, lr))

    loss_rec = []
    loss_pos = []
    loss_neg = []
    loss_grad = []

    for (ir, it), (ir_path, it_path) in tqdm_loader:
        ir = ir.cuda()  # torch.Size([16, 1, 256, 256])
        it = it.cuda()

        ir_affine, affine_disp = affine(ir)
        ir_elastic, elastic_disp = elastic(ir_affine)
        disp = affine_disp + elastic_disp
        ir_warp = ir_elastic

        ir_warp.detach_()
        disp.detach_()

        ir_pred, f_warp, flow, int_flow1, int_flow2, disp_pre = net(it, ir_warp)
        loss1, loss2, grad_loss, loss = _warp_Dense_loss_unsupervised(criterion, ir_pred, f_warp, it, ir_warp, flow,
                                                                      int_flow1, int_flow2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 移除visdom图像显示代码

        loss_rec.append(loss.item())
        loss_pos.append(loss1.item())
        loss_neg.append(loss2.item())
        loss_grad.append(grad_loss.item())

    loss_avg = numpy.mean(loss_rec)
    loss_pos_avg = numpy.mean(loss_pos)
    loss_neg_avg = numpy.mean(loss_neg)
    loss_grad_avg = numpy.mean(loss_grad)

    # 打印损失而不是用visdom绘制
    print(f"Epoch {epoch} - Total Loss: {loss_avg:.6f}, Feats Loss: {loss_pos_avg:.6f}, "
          f"Pixel Loss: {loss_neg_avg:.6f}, Grad Loss: {loss_grad_avg:.6f}")

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def _warp_Dense_loss_unsupervised(criterion, im_pre, im_fwarp, im_fix, im_warp, flow, flow1, flow2):
    total_loss, multi, ncc, grad = criterion(im_pre, im_fwarp, im_fix, im_warp, flow, flow1, flow2)
    return multi, ncc, grad, total_loss

def save_checkpoint(net, epoch, cache):
    model_folder = cache
    model_out_path = str(model_folder / f'cp_{epoch:04d}.pth')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = hyper_args()
    # 移除visdom初始化
    main(args)
