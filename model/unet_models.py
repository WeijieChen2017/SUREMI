""" Full assembly of the parts to form the complete network """
from torch import nn
from .unet_parts import *
from .vit import *
from einops.layers.torch import Rearrange

class UNet_FC(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_FC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.conv1 = DoubleConv(n_channels, 256)
        self.conv2 = DoubleConv(256, 256)
        self.conv3 = DoubleConv(256, 256)
        self.conv4 = DoubleConv(256, 256)
        self.conv5 = DoubleConv(256, 256)
        self.conv6 = DoubleConv(256, 256)
        self.outc = OutConv(256, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.outc(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 256, bilinear)
        self.outc = OutConv(256, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet_simple(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_simple, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.hidden_1 = DoubleConv(1024, 1024)
        self.hidden_2 = DoubleConv(1024, 1024)
        self.up1 = Up_simple(1024, 512, bilinear)
        self.up2 = Up_simple(512, 256, bilinear)
        self.up3 = Up_simple(256, 128, bilinear)
        self.up4 = Up_simple(128, 256, bilinear)
        self.outc = OutConv(256, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits


class UNet_E(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.hidden = DoubleConv(1024, 1024)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return self.hidden(x)

class UNet_D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.hidden = DoubleConv(1024, 1024)
        self.up1 = SimpleUp(1024, 512, bilinear)
        self.up2 = SimpleUp(512, 256, bilinear)
        self.up3 = SimpleUp(256, 128, bilinear)
        self.up4 = SimpleUp(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_bridge(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pre_train=False):
        super(UNet_bridge, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.pre_train = pre_train

        self.inc = DoubleConv(n_channels, 64) # 256
        self.down1 = Down(64, 128) # 256
        self.down2 = Down(128, 256) # 128
        self.down3 = Down(256, 512) # 64
        self.down4 = Down(512, 1024) # 32
        self.hidden_1 = DoubleConv(1024, 1024) # 16

        # -->Input---> torch.Size([10, 3, 256, 256])
        # -->Inc-----> torch.Size([10, 64, 256, 256])
        # -->Down1---> torch.Size([10, 128, 128, 128])
        # -->Down2---> torch.Size([10, 256, 64, 64])
        # -->Down3---> torch.Size([10, 512, 32, 32])
        # -->Down4---> torch.Size([10, 1024, 16, 16])
        # -->Hidden1-> torch.Size([10, 1024, 16, 16])
        # -->Hidden2-> torch.Size([10, 1024, 16, 16])
        # -->Up1-----> torch.Size([10, 512, 32, 32])
        # -->Up2-----> torch.Size([10, 256, 64, 64])
        # -->Up3-----> torch.Size([10, 128, 128, 128])
        # -->Up4-----> torch.Size([10, 64, 256, 256])
        # -->Outc----> torch.Size([10, 1, 256, 256])

        CompFea_lenX, CompFea_lenY = 16, 16
        patch_lenX, patch_lenY = 1, 1
        num_patches = (CompFea_lenX // patch_lenX) * (CompFea_lenY // patch_lenY) # 256
        patch_dim = 1024 * patch_lenX * patch_lenY # 1024
        dim = 1024

        # 10, 1024, 16, 16 -> 10, 256, 1024 -> 10, 256, 1024
        self.embedding = nn.Sequential(
            Rearrange('b c (cfx px) (cfy py) -> b (cfx cfy) (px py c)', px = patch_lenX, py = patch_lenX),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(dim=dim, depth=6, heads=16,
                                       dim_head=64, mlp_dim=1024, dropout=0.1)

        # image_size = 256,
        # patch_size = 32,
        # num_classes = 1000,
        # dim = 1024,
        # depth = 6,
        # heads = 16,
        # mlp_dim = 2048,
        # dropout = 0.1,
        # emb_dropout = 0.1

        #-->embedding---> torch.Size([10, 256, 1024])
        #-->dropout---> torch.Size([10, 256, 1024])
        #-->Bridge---> torch.Size([10, 256, 1024])

        self.unembedding = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange(' b (cfx cfy) (px py c) -> b c (cfx px) (cfy py)', 
                px = patch_lenX, py = patch_lenX,
                cfx = CompFea_lenX, cfy = CompFea_lenY),
        )

        self.hidden_2 = DoubleConv(1024, 1024)
        self.up1 = Up_simple(1024, 512, bilinear)
        self.up2 = Up_simple(512, 256, bilinear)
        self.up3 = Up_simple(256, 128, bilinear)
        self.up4 = Up_simple(128, 256, bilinear)
        self.outc = OutConv(256, n_classes)

        if self.pre_train:
            no_grad_list = [self.inc, self.down1, self.down2, self.down3, self.down4, self.hidden_1,
                            self.hidden_2, self.up1, self.up2, self.up3, self.up4, self.outc]
            for layer in no_grad_list:
                for p in layer.parameters():
                    p.requires_grad = False


    def forward(self, x):

        # with torch.no_grad():
        # print()
        # print("-->Input--->", x.size())
        x = self.inc(x)
        # print("-->Inc--->", x.size())
        x = self.down1(x)
        # print("-->Down1--->", x.size())
        x = self.down2(x)
        # print("-->Down2--->", x.size())
        x = self.down3(x)
        # print("-->Down3--->", x.size())
        x = self.down4(x)
        # print("-->Down4--->", x.size())
        x = self.hidden_1(x)
        # print("-->Hidden1--->", x.size())

        x = self.embedding(x) + self.pos_embedding
        # print("-->embedding--->", x.size())
        x = self.dropout(x)
        # print("-->dropout--->", x.size())
        x = self.transformer(x)
        # print("-->Bridge--->", x.size())
        x = self.unembedding(x)
        # print("-->unembedding--->", x.size())

        # with torch.no_grad():
        x = self.hidden_2(x)
        # print("-->Hidden2--->", x.size())
        x = self.up1(x)
        # print("-->Up1--->", x.size())
        x = self.up2(x)
        # print("-->Up2--->", x.size())
        x = self.up3(x)
        # print("-->Up3--->", x.size())
        x = self.up4(x)
        # print("-->Up4--->", x.size())
        x = self.outc(x)
        # print("-->Outc--->", x.size())
        
        # exit()
        return x

class tf_module_skip(nn.Module):
    def __init__(self, CompFea_len, patch_len, inchannel):
        super(tf_module_skip, self).__init__()
        self.CompFea_len = CompFea_len
        self.patch_len = patch_len

        num_patches = (CompFea_len // patch_len) * (CompFea_len // patch_len)
        patch_dim = inchannel * patch_len * patch_len # 1024
        dim = 1024

        # input is 256x256
        # CompFea_len is 256, patch_len is 16
        # num_patches = 16*16 = 256
        # patch_dim = 1024 * 256
        # dim = 1024
        # b, 64, 256, 256 -> b, 64, 16x16, 16x16 -> b, 16x16, 16x16x64
        # (2560x16384 and 262144x1024)

        self.embedding = nn.Sequential(
            Rearrange('b c (cfx px) (cfy py) -> b (cfx cfy) (px py c)',
                px = patch_len, py = patch_len))
        self.linear1 = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(0.1)
        self.transformer = Transformer(dim=dim, depth=6, heads=16,
                                       dim_head=64, mlp_dim=1024, dropout=0.1)

        self.linear2 = nn.Linear(dim, patch_dim)
        # torch.Size([10, 256, 16384]) -> torch.Size([10, 64, 256, 256])
        self.unembedding = nn.Sequential(
            Rearrange(' b (cfx cfy) (px py c) -> b c (cfx px) (cfy py)', 
                px = patch_len, py = patch_len,
                cfx = CompFea_len // patch_len, cfy = CompFea_len // patch_len))

    def forward(self, x):
        x = self.embedding(x)
        # print(x.size())
        x = self.linear1(x)
        # print(x.size())
        x += self.pos_embedding
        # print(x.size())
        x = self.dropout(x)
        # print(x.size())
        x = self.transformer(x)
        # print(x.size())
        x = self.linear2(x)
        # print(x.size())
        x = self.unembedding(x)
        # print(x.size())
        return x


class UNet_bridge_skip(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pre_train=False):
        super(UNet_bridge_skip, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.pre_train = pre_train
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # inc, down1, down2, down3, down4
        self.tf_config = [[256, 16],[128, 8],[64, 4],[32, 2],[16, 1]]
        self.tf_inc = tf_module_skip(CompFea_len=256, patch_len=16, inchannel=64)
        self.tf_down1 = tf_module_skip(CompFea_len=128, patch_len=8, inchannel=128)
        self.tf_down2 = tf_module_skip(CompFea_len=64, patch_len=4, inchannel=256)
        self.tf_down3 = tf_module_skip(CompFea_len=32, patch_len=2, inchannel=512)
        self.tf_down4 = tf_module_skip(CompFea_len=16, patch_len=1, inchannel=512)

        # -->Input---> torch.Size([10, 3, 256, 256])
        # -->inc---> torch.Size([10, 64, 256, 256])
        # -->down1---> torch.Size([10, 128, 128, 128])
        # -->down2---> torch.Size([10, 256, 64, 64])
        # -->down3---> torch.Size([10, 512, 32, 32])
        # -->down4---> torch.Size([10, 512, 16, 16])
        # -->up1---> torch.Size([10, 256, 32, 32])  <-down3[512]<32>+down4[512]<16>
        # -->up2---> torch.Size([10, 128, 64, 64])  <-up1[256]<32>  +down2[256]<64>
        # -->up3---> torch.Size([10, 64, 128, 128]) <-up2[128]<64>  +down1[128]<128>
        # -->up4---> torch.Size([10, 256, 256, 256])<-up3[64]<128>  +inc[64]<256>
        # -->outc---> torch.Size([10, 1, 256, 256])

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 256, bilinear)
        self.outc = OutConv(256, n_classes)

        if self.pre_train:
            no_grad_list = [self.inc, self.down1, self.down2, self.down3, self.down4,
                            self.up1, self.up2, self.up3, self.up4, self.outc]
            for layer in no_grad_list:
                for p in layer.parameters():
                    p.requires_grad = False


    def forward(self, x):
        # print()
        # print("-->Input--->", x.size())
        x1 = self.inc(x)
        # print("-->inc--->", x1.size())
        x2 = self.down1(x1)
        # print("-->down1--->", x2.size())
        x3 = self.down2(x2)
        # print("-->down2--->", x3.size())
        x4 = self.down3(x3)
        # print("-->down3--->", x4.size())
        x5 = self.down4(x4)
        # print("-->down4--->", x5.size())

        tf_x1 = self.tf_inc(x1)
        # print("-->tf_x1--->", tf_x1.size())
        tf_x2 = self.tf_down1(x2)
        # print("-->tf_x2--->", tf_x2.size())
        tf_x3 = self.tf_down2(x3)
        # print("-->tf_x3--->", tf_x3.size())
        tf_x4 = self.tf_down3(x4)
        # print("-->tf_x4--->", tf_x4.size())
        tf_x5 = self.tf_down4(x5)
        # print("-->tf_x5--->", tf_x5.size())
        
        x = self.up1(tf_x5, tf_x4)
        # print("-->up1--->", x.size())
        x = self.up2(x, tf_x3)
        # print("-->up2--->", x.size())
        x = self.up3(x, tf_x2)
        # print("-->up3--->", x.size())
        x = self.up4(x, tf_x1)
        # print("-->up4--->", x.size())
        x = self.outc(x)
        # print("-->outc--->", x.size())
        # exit()
        return x


class UNet_intra_skip(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_intra_skip, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down_intra(64, 128)
        self.down2 = Down_intra(128, 256)
        self.down3 = Down_intra(256, 512)
        self.down4 = Down_intra(512, 1024)
        self.hidden_1 = DoubleConv(1024, 1024)
        self.hidden_2 = DoubleConv(1024, 1024)
        self.up1 = Up_intra(1024, 512, bilinear)
        self.up2 = Up_intra(512, 256, bilinear)
        self.up3 = Up_intra(256, 128, bilinear)
        self.up4 = Up_intra(128, 256, bilinear)
        self.outc = OutConv(256, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits


class UNet_intra_level_skip(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_intra_skip, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down_intra(64, 128)
        self.down2 = Down_intra(128, 256)
        self.down3 = Down_intra(256, 512)
        self.down4 = Down_intra(512, 1024)
        self.hidden_1 = DoubleConv(1024, 1024)
        self.hidden_2 = DoubleConv(1024, 1024)
        self.up1 = Up_intra(1024, 512, bilinear)
        self.up2 = Up_intra(512, 256, bilinear)
        self.up3 = Up_intra(256, 128, bilinear)
        self.up4 = Up_intra(128, 256, bilinear)
        self.outc = OutConv(256, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits

