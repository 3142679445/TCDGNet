import os
import pandas
import torch
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision.transforms import transforms
from tqdm import tqdm

from net.UNet import UNet
from net.Net import TCDGNet
from tools import metrics
from tools.dataset import Dataset, Datasesloader,crackDataset_no_augmentation
from tools.loss import SoftDiceLoss
from tools.makedir import makedir
from tools.seed import set_seed
from tools.sortresult import sortresult


import setproctitle

processname = "TCDGNet"
setproctitle.setproctitle("%s" % processname)
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)
def train(size, name="Net1", outnum=1, dataset="deepcrack", test=False, gpuid="0", init_lr=0.001):
    set_seed(727392)
    global model, root

    val_gap_num = 1
    val_gap_mod = 0
    init_lr = init_lr

    batchsize = 2
    weight_decay = 0.001
    cfgsize = size
    # pretrain=True
    pretrain = False

    momentum = 0.9
    decay_factor = 0.1

    if dataset == "deepcrack":
        root = r"your dataset path" 

    modelpath = "pre trained model address.pth"
    if test:
        epoch = 1
        savepath = makedir("Weight storage address during test")
    else:
        epoch = 120
        savepath = makedir("Weight storage address during train")
    #writer = SummaryWriter("runs/log")
    with open(savepath + "/" + name + ".txt", 'w') as file:
        file.write(name + "\n" + root + "\n" + str(batchsize)+'\n'+str(init_lr))
    print("path:", savepath)
    criterion = SoftDiceLoss()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    device = torch.device("cuda")

    if name == "Unet":
        model = UNet()
    elif name == "TCDGNet":
        model = TCDGNet()


    if pretrain:
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['net'], strict=False)
        num_loaded_params = len(model.state_dict())
        print(f"加载:{num_loaded_params}")

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=300, gamma=decay_factor)

    trainset = Datasesloader(root, savepath=savepath, txt="train.txt", imgsize=cfgsize)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                               shuffle=True, num_workers=4, drop_last=True)

    testset = Datasesloader(root, savepath=savepath, txt="test.txt", imgsize=cfgsize)
#     testset = crackDataset_no_augmentation(root, savepath=savepath, txt="train.txt", imgsize=cfgsize) #No data augmentation operation
    val_loader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=4, drop_last=True)

    for it in range(epoch):
        model.train()
        lr = scheduler.get_lr()[0]
        loss = 0
        bar1 = tqdm(enumerate(train_loader), total=len(train_loader))
        bar1.set_description('Epoch %d --- Training --- :' % it)
        # for idx, batch in enumerate(train_loader):
        for idx, (img, label) in bar1:
            # img = batch[0].to(device)
            # label = batch[1].to(device)
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            if outnum == 2:
                finalout, out1 = model(img)
                loss1 = criterion(finalout.view(-1, 1), label.view(-1, 1)) / batchsize
                loss2 = criterion(out1.view(-1, 1), label.view(-1, 1)) / batchsize
                output_loss = loss1 + loss2
                loss += output_loss
                output_loss.backward()
                optimizer.step()
            elif outnum == 1:
                pred_output = model(img)
                output_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1)) / batchsize
                loss += output_loss
                output_loss.backward()
                optimizer.step()
        scheduler.step()
        train_aux_loss_mean = (total_aux_loss.item() if 'total_aux_loss' in locals() else 0)
        trainlosslist = [lr, loss.item() / len(train_loader) / batchsize * 100, train_aux_loss_mean]
        print("trainloss", trainlosslist)
    
        data_trainloss = pandas.DataFrame([trainlosslist])
        data_trainloss.to_csv(savepath + '/trainloss.csv', mode='a', header=False, index=False)
    
        # val
        if it % val_gap_num == val_gap_mod:
            bar2 = tqdm(enumerate(val_loader), total=len(val_loader))
            bar2.set_description('Epoch %d --- eval --- :' % it)
            model.eval()
            with torch.no_grad():
                loss = 0
                acc = 0
                precision = 0
                recall = 0
                f1 = 0
                iou = 0
                miou = 0
                for idx, (img, label) in bar2:
                    img = img.to(device)
                    label = label.to(device)
                    if outnum == 2:
                        pred_output, pout1 = model(img)
                        val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1)) + criterion(pout1.view(-1, 1),
                                                                           label.view(-1, 1))
                    elif outnum == 1:
                        pred_output = model(img)
                        val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1)) / batchsize
    
                    loss += val_loss
    
                    pred = torch.sigmoid(pred_output)
                    # pred = pred_output
                    ac, p, r, f, = metrics.f1_loss(label[0], pred)
                    acc += ac
                    precision += p
                    recall += r
                    f1 += f
                    i, _ = metrics.iou_score(pred, label[0])
                    iou += i
                    mi = metrics.miou(pred, label[0])
                    miou += mi
                # [acc,precision,recall,f1,mIoU]
                l = len(val_loader)
                acclist = [acc / l * 100, precision / l * 100, recall / l * 100, f1 / l * 100, iou / l * 100,
                           miou / l * 100, loss.item() / l * 100, it]
    
                vallosslist = [loss.item() / len(val_loader) * 100]
                print("valloss", vallosslist)
                data_valloss = pandas.DataFrame([vallosslist])
                data_valloss.to_csv(savepath + '/valloss.csv', mode='a', header=False, index=False)
    
    
                print("acc", acclist)
                data_acc = pandas.DataFrame([acclist])
                data_acc.to_csv(savepath + '/acc.csv', mode='a', header=False, index=False)
    
            checkpoint = {
                "net": model.state_dict(),
            }
            if not os.path.exists(savepath + "/models"):
                os.mkdir(savepath + "/models")
            # if it>=(epoch-save_model_num):
            torch.save(checkpoint, savepath + "/models/model" + str(it) + ".pth")
        sortresult(savepath + "/acc.csv")

if __name__ == '__main__':
    gpuid = "0"
    # test=True
    test = False
    datasets = ["deepcrack"]
    modelnames=["TCDGNet"]
#     modelname=modelnames[0]
    lrs=[0.001,0.00105,0.0011]
    lr=lrs[0]
    # for lr in lrs:
    outnum = 1
    for dataset in datasets:
        for modelname in modelnames:
            train(512, name=modelname, outnum=outnum, test=test, dataset=datasets[0], gpuid=gpuid, init_lr=lr)




