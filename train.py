import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import esrt
from data import DIV2K, Set5_val, DIV2K_val
from multiprocessing.pool import ThreadPool
import numpy as np
import utils
import skimage.color as sc
import random
from collections import OrderedDict
import datetime
from importlib import import_module
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description="ESRT")
parser.add_argument("--batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="use cuda")
parser.add_argument("--mps", action="store_true", default=False,
                    help="use MPS")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="/data/DIV2K_decoded",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model", type=str, default='ESRT')

args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
mps = args.mps
if mps:
    device = torch.device("mps")
    print("===> use mps")
elif cuda:
    device = torch.device("cuda")
    print("====> use cuda")
else:
    device = torch.device("cpu")
    print("===> use cpu")

print("===> Loading datasets")

trainset = DIV2K.div2k(args)
#testset = Set5_val.DatasetFromFolderVal("Test_Datasets/Set5/",
#                                       "Test_Datasets/Set5_LR/x{}/".format(args.scale),
#                                       args.scale)
testset = DIV2K_val.DIV2KValidSet("data/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/",
                                  "data/DIV2K/DIV2K_valid_HR/DIV2K_valid_LR_bicubic/X{}/".format(args.scale),
                                  args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)
print("testing loader length: ", len(testing_data_loader))

print("===> Building models")
args.is_train = True


# Displaying 5 random train set images
# for i in range(5):
#     idx = random.randint(0, len(trainset)-1)
#     lr_tensor, hr_tensor = trainset[idx]

#     # Debugging information
#     print("LR Tensor Min:", lr_tensor.min().item())
#     print("LR Tensor Max:", lr_tensor.max().item())
#     print("HR Tensor Min:", hr_tensor.min().item())
#     print("HR Tensor Max:", hr_tensor.max().item())
#     print("LR shape: ", lr_tensor.shape)
#     print("HR shape: ", hr_tensor.shape)

#     # Convert to numpy for visualization
#     lr_array = lr_tensor.detach().cpu().numpy()
#     hr_array = hr_tensor.detach().cpu().numpy()

#     # Plot lr_tensor and hr_tensor
#     fig, axes = plt.subplots(1, 2)
#     axes[0].imshow(lr_array.transpose(1, 2, 0))
#     axes[0].set_title('lr_tensor')
#     axes[1].imshow(hr_array.transpose(1, 2, 0))
#     axes[1].set_title('hr_tensor')
#     plt.show()

model = esrt.ESRT(upscale = args.scale)#architecture.IMDN(upscale=args.scale)

l1_criterion = nn.L1Loss()
l2_criterion = nn.MSELoss()
gamma = 0.1

print("===> Setting GPU")
if cuda or mps:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

if args.pretrained:

    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args.lr)

def display_tensors(lr_tensor, hr_tensor):
    # Check if the inputs are PyTorch tensors
    if isinstance(lr_tensor, torch.Tensor):
        lr_array = lr_tensor.detach().cpu().numpy()
    elif isinstance(lr_tensor, np.ndarray):
        lr_array = lr_tensor
    else:
        raise TypeError("lr_tensor must be either a PyTorch tensor or a NumPy array")
    
    if isinstance(hr_tensor, torch.Tensor):
        hr_array = hr_tensor.detach().cpu().numpy()
    elif isinstance(hr_tensor, np.ndarray):
        hr_array = hr_tensor
    else:
        raise TypeError("hr_tensor must be either a PyTorch tensor or a NumPy array")
    
    # Plot lr_tensor and hr_tensor
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(lr_array[0].transpose(1, 2, 0))
    axes[0].set_title('lr_tensor')
    axes[1].imshow(hr_array[0].transpose(1, 2, 0))
    axes[1].set_title('hr_tensor')
    plt.show()

def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda or args.mps:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        #print("LR tensor shape: ", lr_tensor.shape)
        #print("HR tensor shape: ", hr_tensor.shape)
        #Display the lr_tensor and hr_tensor on the same plot
        
        sr_tensor = model(lr_tensor)
        #display_tensors(sr_tensor, hr_tensor)
        loss = (1-gamma) * l1_criterion(sr_tensor, hr_tensor) + gamma * l2_criterion(sr_tensor, hr_tensor)
        loss_sr = loss

        loss_sr.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss.item()))
            
def forward_chop(model, x, scale, shave=10, min_size=60000):
    print("Starting forward chop")
    # scale = scale#self.scale[self.idx_scale]
    n_GPUs = 1#min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    print("b, c, h, w: ", b, c, h, w)
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    print("h_half, w_half, h_size, w_size: ", h_half, w_half, h_size, w_size)
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]
    #print("lr_list: ", lr_list)
    print("Starting forward")
    if w_size * h_size < min_size:
        print("size is small")
        sr_list = []
        for i in range(0, 4, n_GPUs):
            print("starting iteration: ", i)
            print("Concatenating")
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            print("starting model forward pass")
            sr_batch = model(lr_batch)
            print("Finished model forward pass")
            print("Exending list")
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, scale=args.scale, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]
    print("End forward")
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale
    print("Setting output")
    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def valid(scale):
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    print("===> Validating")
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda or args.mps:
            print('Use gpu')
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
        print("Starting forward")
        with torch.no_grad():
            pre = forward_chop(model, lr_tensor, scale)#model(lr_tensor)
        print("End forwardchop")
        print("transforming to numpy")
        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        print("End transforming")
        crop_size = args.scale
        print("Shaving")
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        print("End shaving")
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        # print(im_pre.shape)
        # print(im_label.shape)
        print("Displaying images")
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(im_pre, cmap='gray')
        axes[0].set_title('im_pre')
        axes[1].imshow(im_label, cmap='gray')
        axes[1].set_title('im_label')
        plt.show()
        print("computing psnr")
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        print("computing ssim")
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))


def save_checkpoint(epoch):
    model_folder = "experiment/checkpoint_ESRT_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

print("===> Training")
print_network(model)
code_start = datetime.datetime.now()
timer = utils.Timer()
for epoch in range(args.start_epoch, args.nEpochs + 1):
    t_epoch_start = timer.t()
    epoch_start = datetime.datetime.now()
    #valid(args.scale)
    train(epoch)
    if epoch%10==0:
        save_checkpoint(epoch)
    epoch_end = datetime.datetime.now()
    print('Epoch cost times: %s' % str(epoch_end-epoch_start))
    t = timer.t()
    prog = (epoch-args.start_epoch+1)/(args.nEpochs + 1 - args.start_epoch + 1)
    t_epoch = utils.time_text(t - t_epoch_start)
    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
    print('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
code_end = datetime.datetime.now()
print('Code cost times: %s' % str(code_end-code_start))
