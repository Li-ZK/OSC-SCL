import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from utils.meter import F1ScoreMeter
from utils.scheduler import WarmupCosineLR
from utils.contrast import contrast_mse

class SPCModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPCModuleIN, self).__init__()
                
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7,1,1), stride=(2,1,1), bias=False)

    def forward(self, input):
        
        input = input.unsqueeze(1)
        
        out = self.s1(input)
        
        return out.squeeze(1) 

class SPAModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, k=49):
        super(SPAModuleIN, self).__init__()

        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k,3,3), bias=False)

    def forward(self, input):
                
        out = self.s1(input)
        out = out.squeeze(2)
        
        return out

class ResSPC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResSPC, self).__init__()
                
        self.spc1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm3d(in_channels),)
        
        self.spc2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),)
        
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
                
        out = self.spc1(input)
        out = self.bn2(self.spc2(out))
        
        return F.leaky_relu(out + input)

class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResSPA, self).__init__()
                
        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm2d(in_channels),)
        
        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
                
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))
        
        return F.leaky_relu(out + input)

class SSRN(nn.Module):
    def __init__(self, num_classes=9, k=49):
        super(SSRN, self).__init__()

        self.layer1 = SPCModuleIN(1, 28)
        self.bn1 = nn.BatchNorm3d(28)
        
        self.layer2 = ResSPC(28,28)
        
        self.layer3 = ResSPC(28,28)
        
        self.layer4 = SPAModuleIN(28, 28, k=k)
        self.bn4 = nn.BatchNorm2d(28)
        
        self.layer5 = ResSPA(28, 28)
        self.layer6 = ResSPA(28, 28)

        self.fc = nn.Linear(28, num_classes)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.layer1(x)))
        x = self.layer2(x)
        x = self.layer3(x)
        spe = F.avg_pool3d(x, [*x.shape[2:]]).flatten(1)

        x = self.bn4(F.leaky_relu(self.layer4(x)))

        x = self.layer5(x)
        x = self.layer6(x)
        spa = F.avg_pool2d(x, x.shape[-1]).squeeze()

        x = self.fc(spa)

        return {
            'spe': spe,
            'spa': spa,
            'logits': x,
        }

class OSC_SCL(pl.LightningModule):
    def __init__(self, args, info, num_classes, anchor_weight=10.0, alpha=1.0, beta=0.5, ksai=2.0):
        super(OSC_SCL, self).__init__()

        self.args = args
        self.info = info
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.ksai = ksai

        self.contrast_computer: callable = contrast_mse

        self.encoder = SSRN(num_classes=num_classes, k=int((self.info.bands_num-7)/2) + 1)

        self.anchor = nn.Parameter(torch.eye(num_classes) * anchor_weight, requires_grad=False)

        self.train_acc_meter = Accuracy()
        self.oa_meter = Accuracy()
        self.aa_meter = MulticlassAccuracy(num_classes + 1, average=None)
        self.auc = AUROC(pos_label=1)
        self.f1_meter = F1ScoreMeter(2)

    def forward(self, x):
        encoder_out = self.encoder(x)
        distance = self.distance_classifier(encoder_out['logits'])

        return {
            **encoder_out,
            'distance': distance,
        }

    def distance_classifier(self, x):

        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d)
        anchor = self.anchor.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x-anchor, 2, 2)

        return dists

    def CACLoss(self, distances, gt):
        device = next(self.parameters()).device
        true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
        non_gt = torch.Tensor([[i for i in range(self.num_classes) if gt[x] != i] for x in range(len(distances))]).long().to(device)
        others = torch.gather(distances, 1, non_gt)

        anchor = torch.mean(true)

        tuplet = torch.exp(-others + true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

        total = self.alpha * anchor + tuplet

        return {
            'total': total,
            'anchor': anchor,
            'tuplet': tuplet
        }

    def training_step(self, batch, idx_batch):
        x, y = batch

        out = self(x)
        y_hat = F.softmax(out['logits'], -1)

        loss_contrast = self.contrast_computer(out['spe'], y, self.ksai) + self.contrast_computer(out['spa'], y, self.ksai)

        loss_cac: dict = self.CACLoss(out['distance'], y)

        loss = loss_cac['total'] + self.beta * loss_contrast

        self.train_acc_meter.update(y_hat, y)
        self.log('loss', loss)

        return loss

    def training_epoch_end(self, output):
        self.log('train_acc', self.train_acc_meter, prog_bar=True)

    def test_step(self, batch, idx_batch):
        x, y = batch
        bi_y = torch.zeros_like(y, dtype=torch.long)
        bi_y[y == self.num_classes] = 1

        out = self(x)
        distance = out['distance']

        y_hat = F.softmax(out['logits'], -1)
        prob, prediction = torch.max(y_hat, -1)

        gamma = distance * (1 - F.softmin(distance))
        score = torch.min(gamma, 1)[0]
        prediction[score > 0.1] = self.num_classes

        bi_prediction = torch.zeros_like(y, dtype=torch.long)
        bi_prediction[score > 0.1] = 1

        self.oa_meter.update(prediction, y)
        self.aa_meter.update(prediction, y)
        self.f1_meter.add(bi_prediction, bi_y)
        self.auc.update(score, bi_y)

    def test_epoch_end(self, outputs):
        print('test_oa:', self.oa_meter.compute().item())
        print('test_aa:', self.aa_meter.compute().mean().item())
        print('f1_micro:', self.f1_meter.value('micro').item())
        print('test_auc:', self.auc.compute().item())
        print('test_classes_acc: ', self.aa_meter.compute().tolist())

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = WarmupCosineLR(optimizer, lr_min=1e-6, lr_max=1e-3, warm_up=100, T_max=self.trainer.max_epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def getMeanAnchor(self, dataloader):
        logits_list = []
        labels_list = []

        self.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                out = self(x)
                logits = out['logits']

                y_hat = F.softmax(logits, -1)
                prediction = y_hat.argmax(1)

                mask = prediction == y
                logits = logits[mask]
                labels = y[mask]

                logits_list += logits.detach().tolist()
                labels_list += labels.tolist()

        logits_list = torch.tensor(logits_list)
        labels_list = torch.tensor(labels_list)

        means = [None for i in range(self.num_classes)]

        for cl in range(self.num_classes):
            x = logits_list[labels_list == cl]
            x = torch.squeeze(x)
            means[cl] = torch.mean(x, 0).detach().tolist()

        return torch.tensor(means)

    def setAnchor(self, anchor):
        self.anchor = nn.Parameter(anchor, requires_grad=False)

def get_model(args, data_info):
    model_args = {
        'args': args,
        'info': data_info,
        'num_classes': len(args.known_classes),
        'alpha': args.alpha,
        'beta': args.beta,
        'ksai': args.ksai,
        'anchor_weight': args.anchor_weight
    }
    model = OSC_SCL(**model_args)
    return model

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, default='OSC_SCL')
    parser.add_argument('--train_num', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--patch', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dataset', choices=['PaviaU', 'Houston13'], default='PaviaU')
    parser.add_argument('--alpha', type=float, default=1.5)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--anchor_weight', type=float, default=10.0)

    args = parser.parse_args()

    if args.dataset == 'PaviaU':
        args.known_classes = [1,2,3,4,5,6,7,8]
        args.unknown_classes = [9,]
        args.ksai = 3.5
    elif args.dataset == 'Houston13':
        args.known_classes = [1,2,3,4,5,7,8,9,10,11,12]
        args.unknown_classes = [6,13,14,15]
        args.ksai = 2.0

    return args
