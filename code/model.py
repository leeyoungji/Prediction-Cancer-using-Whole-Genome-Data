import torch
import torch.nn as nn
import torch.nn.functional as F

#44900
SEQ_LEN = 44900 # maximum length of an input sequence
BATCH_SIZE = 8
EMBED_SIZE = 16
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1 # KERNEL_SIZE - 2
# LEARNING_RATE = 0.01
# WEIGHT_DECAY = 1e-4
VERBOSE = False
SAVE_EVERY = 10

PAD = "<PAD>" # padding
PAD_IDX = 0

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.k = 64
        #64
        self.conv1 = nn.Conv1d(args.num_features,128, kernel_size=3, stride=1, padding=1)

        self.res_blocks = nn.Sequential(  # residual blocks
            res_block(128, 128),
            res_block(128, 128, "vgg"),
            res_block(128, 128),
            res_block(128, 128, "vgg"),
            res_block(128, 128),
            res_block(128, 128)
        )
        self.fc = nn.Sequential(  # fully connected layers
            nn.Linear(128 * self.k, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 12)
        )

        # self.fc = nn.Sequential(  # fully connected layers
        #     nn.Linear(512 * self.k, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 15)
        # )

        self.softmax = nn.Softmax()



    def forward(self, x):
        # x = self.embed(x)  # embedding
        # x = x.transpose(1, 2)  # [batch_size (N), num_feature_maps (D), seq_len (L)]
        x = self.conv1(x)
        # h = self.conv(x)  # temporal convolution
        h = self.res_blocks(x)  # residual blocks
        h = h.topk(self.k)[0].view(BATCH_SIZE, -1)  # k-max pooling
        h = self.fc(h)  # fully connected layers
        y = self.softmax(h)
        return y

class res_block(nn.Module):  # residual block
    def __init__(self, in_channels, out_channels, downsample=None):
        super().__init__()
        first_stride = 2 if downsample == "resnet" else 1
        pool_stride = 2 if downsample else 1

        # architecture
        self.conv_block = conv_block(in_channels, out_channels, first_stride)
        self.pool = None
        if downsample == "kmax":  # k-max pooling (Kalchbrenner et al 2014)
            self.pool = lambda x: x.topk(x.size(2) // 2)[0]
        elif downsample == "vgg":  # VGG-like
            self.pool = nn.MaxPool1d(KERNEL_SIZE, pool_stride, PADDING)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1, pool_stride)

    def forward(self, x):
        y = self.conv_block(x)
        if self.pool:
            y = self.pool(y)
        y += self.shortcut(x)  # ResNet shortcut connections
        return y

class conv_block(nn.Module):  # convolutional block
    def __init__(self, in_channels, out_channels, first_stride):
        super().__init__()

        # architecture
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, KERNEL_SIZE, first_stride, PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
            )

    def forward(self, x):
        return self.sequential(x)

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]

def argmax(x):
    return scalar(torch.max(x, 0)[1])  # for 1D tensor
