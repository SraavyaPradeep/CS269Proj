import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util import conv, predict_flow, deconv, crop_like, correlate


__all__ = ["flownetc", "flownetc_bn"]

class FlowNetC(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetC, self).__init__()

        self.batchNorm = batchNorm
        self.conv_redir = conv(self.batchNorm, 64, 32, kernel_size=1, stride=1)

        self.conv3_1 = conv(self.batchNorm, 473, 256)  # Adjust input channels for concatenation
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        # self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        # self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        # self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        # self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(98)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.upsample = nn.Upsample(size=(900, 1600), mode='bilinear', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
        


    def correlate(input1, input2):
        out_corr = spatial_correlation_sample(
            input1,
            input2,
            kernel_size=1,
            patch_size=21,
            stride=1,
            padding=0,
            dilation_patch=2,
        )
        # collate dimensions 1 and 2 in order to be treated as a
        # regular 4D tensor
        b, ph, pw, h, w = out_corr.size()
        out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
        return F.leaky_relu_(out_corr, 0.1)

    def forward(self, x1, x2):

            out_conv_redir1 = self.upsample(self.conv_redir(x1))
            # print(out_conv_redir1.shape)
            out_conv_redir2 = self.upsample(self.conv_redir(x2))

            out_correlation = correlate(out_conv_redir1, out_conv_redir2)
            in_conv3_1 = torch.cat([out_conv_redir1, out_correlation], dim=1)
            
            out_conv3 = self.conv3_1(in_conv3_1)
            
            out_conv4 = self.conv4_1(self.conv4(out_conv3))
            out_conv5 = self.conv5_1(self.conv5(out_conv4))
            # out_conv6 = self.conv6_1(self.conv6(out_conv5))

            # print(out_conv3.shape, out_conv4.shape, out_conv5.shape, out_conv6.shape)

            # flow6 = self.predict_flow6(out_conv6)
            # print(flow6.shape)
            # flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
            # out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

            # concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
            flow5 = self.predict_flow5(out_conv5)
            flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
            out_deconv4 = crop_like(self.deconv4(out_conv5), out_conv4)

            concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
            flow4 = self.predict_flow4(concat4)
            flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
            out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

            concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
            flow3 = self.predict_flow3(concat3)
            flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv_redir1)
            out_deconv2 = crop_like(self.deconv2(concat3), out_conv_redir1)
            
            concat2 = torch.cat((out_conv_redir1, out_deconv2, flow3_up), 1)
            flow2 = self.predict_flow2(concat2)

            # return flow2_up

            if self.training:
                return flow2, flow3, flow4, flow5
            else:
                return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]


def flownetc(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=False)
    if data is not None:
        model.load_state_dict(data["state_dict"])
    return model


def flownetc_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=True)
    if data is not None:
        model.load_state_dict(data["state_dict"])
    return model
