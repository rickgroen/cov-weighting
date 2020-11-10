from networks.blocks import *


class ResNet50MD(nn.Module):
    """
        Original MonoDepth implementation of ResNet50.
    """

    def __init__(self, normalize=None, do_multi_objective=False):
        super(ResNet50MD, self).__init__()
        # encoder
        self.conv1 = Conv(3, 64, 7, 2, normalize=normalize)     # H/2  -   64D
        self.pool1 = MaxPool(3)                                             # H/4  -   64D
        self.conv2 = ResBlock(64, 64, 3, 2, normalize=normalize)            # H/8  -  256D
        self.conv3 = ResBlock(256, 128, 4, 2, normalize=normalize)          # H/16 -  512D
        self.conv4 = ResBlock(512, 256, 6, 2, normalize=normalize)          # H/32 - 1024D
        self.conv5 = ResBlock(1024, 512, 3, 2, normalize=normalize)         # H/64 - 2048D

        # decoder
        self.upconv6 = Upconv(2048, 512, 3, 2, normalize=normalize)
        self.iconv6 = Conv(1024 + 512, 512, 3, 1, normalize=normalize)

        self.upconv5 = Upconv(512, 256, 3, 2, normalize=normalize)
        self.iconv5 = Conv(512+256, 256, 3, 1, normalize=normalize)

        self.upconv4 = Upconv(256, 128, 3, 2, normalize=normalize)
        self.iconv4 = Conv(256 + 128, 128, 3, 1, normalize=normalize)
        self.disp4_layer = GetDisp(128, num_out_layers=2)

        self.upconv3 = Upconv(128, 64, 3, 2, normalize=normalize)
        self.iconv3 = Conv(64 + 64 + 2, 64, 3, 1, normalize=normalize)
        self.disp3_layer = GetDisp(64, num_out_layers=2)

        self.upconv2 = Upconv(64, 32, 3, 2, normalize=normalize)
        self.iconv2 = Conv(64 + 32 + 2, 32, 3, 1, normalize=normalize)
        self.disp2_layer = GetDisp(32, num_out_layers=2)

        self.upconv1 = Upconv(32, 16, 3, 2, normalize=normalize)
        self.iconv1 = Conv(16 + 2, 16, 3, 1, normalize=normalize)
        self.disp1_layer = GetDisp(16, num_out_layers=2)

        self.return_shared_output = do_multi_objective

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)

        if self.return_shared_output:
            # Specifically for Multi-objective optimization.
            return (skip2, skip1, iconv4), (self.disp1, self.disp2, self.disp3, self.disp4)
        return self.disp1, self.disp2, self.disp3, self.disp4


class ResNet18MD(nn.Module):
    def __init__(self, normalize=None, do_multi_objective=False):
        super(ResNet18MD, self).__init__()
        # encoder
        self.conv1 = Conv(3, 64, 7, 2, normalize=normalize)                 # H/2  -   64D
        self.pool1 = MaxPool(3)                                             # H/4  -   64D
        self.conv2 = ResBlockBasic(64, 64, 2, 2, normalize=normalize)       # H/8  -   64D
        self.conv3 = ResBlockBasic(64, 128, 2, 2, normalize=normalize)      # H/16 -  128D
        self.conv4 = ResBlockBasic(128, 256, 2, 2, normalize=normalize)     # H/32 -  256D
        self.conv5 = ResBlockBasic(256, 512, 2, 2, normalize=normalize)     # H/64 -  512D

        # decoder
        self.upconv6 = Upconv(512, 512, 3, 2, normalize=normalize)
        self.iconv6 = Conv(256+512, 512, 3, 1, normalize=normalize)

        self.upconv5 = Upconv(512, 256, 3, 2, normalize=normalize)
        self.iconv5 = Conv(128 + 256, 256, 3, 1, normalize=normalize)

        self.upconv4 = Upconv(256, 128, 3, 2, normalize=normalize)
        self.iconv4 = Conv(64 + 128, 128, 3, 1, normalize=normalize)
        self.disp4_layer = GetDisp(128, num_out_layers=2)

        self.upconv3 = Upconv(128, 64, 3, 2, normalize=normalize)
        self.iconv3 = Conv(64 + 64 + 2, 64, 3, 1, normalize=normalize)
        self.disp3_layer = GetDisp(64, num_out_layers=2)

        self.upconv2 = Upconv(64, 32, 3, 2, normalize=normalize)
        self.iconv2 = Conv(64 + 32 + 2, 32, 3, 1, normalize=normalize)
        self.disp2_layer = GetDisp(32, num_out_layers=2)

        self.upconv1 = Upconv(32, 16, 3, 2, normalize=normalize)
        self.iconv1 = Conv(16 + 2, 16, 3, 1, normalize=normalize)
        self.disp1_layer = GetDisp(16, num_out_layers=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

        self.return_shared_output = do_multi_objective

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)

        if self.return_shared_output:
            # Specifically for Multi-objective optimization.
            return (skip2, skip1, iconv4), (self.disp1, self.disp2, self.disp3, self.disp4)
        return self.disp1, self.disp2, self.disp3, self.disp4
