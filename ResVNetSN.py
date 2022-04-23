import torch
import torch.nn as nn
import torch.nn.functional as F
import SwitchNorm as sn


class SingleLayer(nn.Module):
    def __init__(self, num_features, drop_rate=0, dilation_rate=1):
        super(SingleLayer, self).__init__()
        self.conv = nn.Conv3d(num_features, num_features, kernel_size=3, dilation=dilation_rate, padding=dilation_rate, bias=False)
        self.sn = sn.SwitchNorm3d(num_features)
        self.relu = nn.PReLU(num_features)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.relu(self.sn(self.conv(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        return out


class ResBlock(nn.Module):
    def __init__(self, num_features, num_layers, drop_rate=0, dilation_rate=1):
        super(ResBlock, self).__init__()
        layers = []
        for i in range(int(num_layers)):
            layers.append(SingleLayer(num_features, drop_rate, dilation_rate=dilation_rate))
        self.layers = nn.Sequential(*layers)
        self.relu = nn.PReLU(num_features)

    def forward(self, x):
        out = self.layers(x)
        out = self.relu(torch.add(out, x))
        return out


class DownTrans(nn.Module):
    def __init__(self, num_input_features, num_output_features, drop_rate=0):
        super(DownTrans, self).__init__()

        self.down_conv = nn.Conv3d(num_input_features, num_output_features, kernel_size=2, stride=2, bias=False)
        self.sn = sn.SwitchNorm3d(num_output_features)

        self.relu = nn.PReLU(num_output_features)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.relu(self.sn(self.down_conv(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        return out


class UpTrans(nn.Module):
    def __init__(self, num_input_features, num_out_features, drop_rate=0):
        super(UpTrans, self).__init__()

        self.up_conv = nn.ConvTranspose3d(num_input_features, num_out_features // 2, kernel_size=2, stride=2, bias=False)
        self.sn = sn.SwitchNorm3d(num_out_features // 2)

        self.relu = nn.PReLU(num_out_features // 2)

        self.drop_rate = drop_rate

    def forward(self, x, skip):
        out = self.relu(self.sn(self.up_conv(x)))
        out = torch.cat((out, skip), 1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        return out


class Input(nn.Module):
    def __init__(self, num_out_features):
        super(Input, self).__init__()
        self.conv = nn.Conv3d(1, num_out_features, kernel_size=5, padding=2, bias=False)
        self.sn = sn.SwitchNorm3d(num_out_features)
        self.relu = nn.PReLU(num_out_features)

    def forward(self, x):
        out = self.sn(self.conv(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 1)
        out = self.relu(torch.add(out, x16))
        return out


class Output(nn.Module):
    def __init__(self, num_input_features):
        super(Output, self).__init__()
        self.conv1 = nn.Conv3d(num_input_features, 2, kernel_size=5, padding=2, bias=False)
        self.sn1 = sn.SwitchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = nn.PReLU(2)

        self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.sn1(self.conv1(x)))
        out = self.conv2(out)
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
#        print(out.size())
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out, 1)
        # treat channel 0 as the predicted output
        return out


class ResVNet(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)

        self.up4 = UpTrans(num_init_features*16, num_init_features*16, drop_rate=0.2)
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.up3 = UpTrans(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features*2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)

        up4 = self.up4(resdown4, resdown3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out


class ASPP_module(nn.Module):
    def __init__(self, num_input_features, num_out_features, dilation_rates):
        super(ASPP_module, self).__init__()

        self.input = nn.Sequential(sn.SwitchNorm3d(num_input_features), nn.ReLU(inplace=True))

        self.conv11_0 = nn.Sequential(nn.Conv3d(num_input_features, num_out_features, kernel_size=1,
                                                padding=0, dilation=dilation_rates[0], bias=False),
                                      sn.SwitchNorm3d(num_out_features))
        self.conv33_1 = nn.Sequential(nn.Conv3d(num_input_features, num_out_features, kernel_size=3,
                                                padding=dilation_rates[1], dilation=dilation_rates[1], bias=False),
                                      sn.SwitchNorm3d(num_out_features))
        self.conv33_2 = nn.Sequential(nn.Conv3d(num_input_features, num_out_features, kernel_size=3,
                                                padding=dilation_rates[2], dilation=dilation_rates[2], bias=False),
                                      sn.SwitchNorm3d(num_out_features))
        self.conv33_3 = nn.Sequential(nn.Conv3d(num_input_features, num_out_features, kernel_size=3,
                                                padding=dilation_rates[3], dilation=dilation_rates[3], bias=False),
                                      sn.SwitchNorm3d(num_out_features))

        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv_avg = nn.Conv3d(num_input_features, num_out_features, kernel_size=1, bias=False)

        self.num_input_features = num_input_features
        self.num_out_features = num_out_features

        self.cat_conv = nn.Conv3d(num_out_features * 5, num_out_features, kernel_size=1, bias=False)

    def forward(self, x):

        input = self.input(x)

        conv11_0 = self.conv11_0(input)
        conv33_1 = self.conv33_1(input)
        conv33_2 = self.conv33_2(input)
        conv33_3 = self.conv33_3(input)

        avg = self.global_avg_pool(input)
        conv_avg = self.conv_avg(avg)
        upsample = F.upsample(conv_avg, size=x.size()[2:], mode='trilinear', align_corners=True)

        concate = torch.cat((conv11_0, conv33_1, conv33_2, conv33_3, upsample), 1)

        return self.cat_conv(concate)


class ResVNet_ASPP(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1],
                 dilation_rates=[1, 6, 12, 18]):
        super(ResVNet_ASPP, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], dilation_rate=2, drop_rate=0.2)

        self.aspp = ASPP_module(num_init_features*8, num_init_features*8, dilation_rates)

        self.up3 = UpTrans(num_init_features*8, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], dilation_rate=2, drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features*2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        aspp = self.aspp(resdown3)

        up3 = self.up3(aspp, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out


class DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, growth_rate, dilation_rate, drop_out):
        super(DenseAsppBlock, self).__init__()

        interChannels = 4 * growth_rate

        self.sn1 = sn.SwitchNorm3d(input_num)
        self.conv1 = nn.Conv3d(input_num, interChannels, kernel_size=1, bias=False)

        self.sn2 = sn.SwitchNorm3d(interChannels)
        self.conv2 = nn.Conv3d(interChannels, growth_rate, kernel_size=3, dilation=dilation_rate, padding=dilation_rate, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.drop_rate = drop_out

    def forward(self, x):

        out = self.conv1(self.relu(self.sn1(x)))
        out = self.conv2(self.relu(self.sn2(out)))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)

        out = torch.cat((x, out), 1)

        return out


class DenseASPP_module(nn.Module):
    def __init__(self, num_input_features, dense_growth_rate, dilation_rates):
        super(DenseASPP_module, self).__init__()

        num_features = num_input_features
        self.ASPP_3 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[0], drop_out=0.2)
        num_features += dense_growth_rate

        self.ASPP_6 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[1], drop_out=0.2)
        num_features += dense_growth_rate

        self.ASPP_12 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[2], drop_out=0.2)
        num_features += dense_growth_rate

        self.ASPP_18 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[3], drop_out=0.2)
        num_features += dense_growth_rate

        self.ASPP_24 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[4], drop_out=0.2)
        num_features += dense_growth_rate

        self.trans = nn.Conv3d(num_features, num_input_features, kernel_size=1, bias=False)

    def forward(self, x):

        aspp3 = self.ASPP_3(x)
        aspp6 = self.ASPP_6(aspp3)
        aspp12 = self.ASPP_12(aspp6)
        aspp18 = self.ASPP_18(aspp12)
        aspp24 = self.ASPP_24(aspp18)

        return self.trans(aspp24)


class ResVNet_DenseASPP(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1],
                 dilation_rates=[3, 6, 12, 18, 24], dense_growth_rate=64):
        super(ResVNet_DenseASPP, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], dilation_rate=2, drop_rate=0.2)

        self.denseaspp = DenseASPP_module(num_init_features*8, dense_growth_rate, dilation_rates)

        self.up3 = UpTrans(num_init_features*8, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], dilation_rate=2, drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features*2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        denseaspp = self.denseaspp(resdown3)

        up3 = self.up3(denseaspp, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out