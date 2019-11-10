# TumorNet without source; Downsample by pooling; activation is relu
# Which can use transfer lerning directly
import torch
from torch import nn
import numpy as np
import SimpleITK as sitk
from medpy import metric

def saved_preprocessed(savedImg,origin,direction,xyz_thickness,saved_name):
	origin = tuple(k.item() for k in origin)
	direction = tuple(k.item() for k in direction)
	xyz_thickness = tuple(k.item() for k in xyz_thickness)
	savedImg = np.squeeze(np.argmax(savedImg.detach().cpu().numpy(),axis=1),0).astype(np.float32)
	newImg = sitk.GetImageFromArray(savedImg)
	newImg.SetOrigin(origin)
	newImg.SetDirection(direction)
	newImg.SetSpacing(xyz_thickness)
	sitk.WriteImage(newImg, saved_name)

def Dice(output2, target):
	pred_lesion = np.argmax(output2.detach().cpu().numpy(), axis=1)
	target = np.squeeze(target.detach().cpu().numpy(), axis=1)
	true_lesion = target == 2
	# Compute per-case (per patient volume) dice.
	if not np.any(pred_lesion) and not np.any(true_lesion):
		tumor_dice = 1.
		print('tumor_dice = 1')
	else:
		tumor_dice = metric.dc(pred_lesion, true_lesion)
	return tumor_dice

def one_hot(scores, labels):
    _labels = torch.zeros_like(scores)
    _labels.scatter_(dim=1, index=labels.long(), value=1)#scatter_(input, dim, index, src)
    return _labels

class DiceLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.smooth = 1e-5

	def forward(self, output2, target):
		temp = target.clone() # deep clone
		temp[target == 2] = 1
		temp[target <= 1] = 0
		target2 = one_hot(output2, temp)
		intersection2 = 2. * (output2 * target2).sum()
		denominator2 = output2.sum() + target2.sum()
		dice2 = (intersection2 + self.smooth) / (denominator2 + self.smooth)
		dice = 1 - dice2
		return dice

class PostRes(nn.Module):
	def __init__(self, n_in, n_out, stride = 1):
		super(PostRes, self).__init__()
		self.resBlock = nn.Sequential(
			nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1),
			nn.InstanceNorm3d(n_out),
			nn.ReLU(inplace=True),
			# nn.PReLU(),
			nn.Conv3d(n_out, n_out, kernel_size=3, padding=1),
			nn.InstanceNorm3d(n_out)
		)
		self.relu = nn.ReLU(inplace=True)
		# self.prelu = nn.PReLU()

		if stride != 1 or n_out != n_in:
			self.shortcut = nn.Sequential(
				nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
				nn.InstanceNorm3d(n_out))
		else:
			self.shortcut = None

	def forward(self, x):
		residual = x
		if self.shortcut is not None:
			residual = self.shortcut(x)

		out = self.resBlock(x)
		out += residual
		out = self.relu(out)
		# out = self.prelu(out)
		return out

class Decoder2(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_blocks_back = [3, 3, 2, 2]  # [5-2]
		self.nff = [1, 8, 16, 32, 64, 128]  # NumFeature_Forw[0-5]
		self.nfb = [64, 32, 16, 8, 2]  # NunFeaturn_Back[5-0]
		#deconv4-1,output
		self.deconv4 = nn.Sequential(
			nn.ConvTranspose3d(self.nff[5], self.nfb[0], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[0]),
			nn.ReLU(inplace=True)
			# nn.PReLU()
        )
		self.deconv3 = nn.Sequential(
			nn.ConvTranspose3d(self.nfb[0], self.nfb[1], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[1]),
			nn.ReLU(inplace=True)
			# nn.PReLU()
        )
		self.deconv2 = nn.Sequential(
			nn.ConvTranspose3d(self.nfb[1], self.nfb[2], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[2]),
			nn.ReLU(inplace=True)
			# nn.PReLU()
        )
		self.deconv1 = nn.Sequential(
			nn.ConvTranspose3d(self.nfb[2], self.nfb[3], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[3]),
			nn.ReLU(inplace=True)
			# nn.PReLU()
        )
		self.output = nn.Sequential(
			nn.Conv3d(self.nfb[3], self.nfb[3], kernel_size=1),
			nn.InstanceNorm3d(self.nfb[3]),
		    nn.ReLU(inplace=True),
			# nn.PReLU(),
			# nn.Dropout3d(p = 0.3),
		    nn.Conv3d(self.nfb[3], self.nfb[4], kernel_size=1))  # since class number = 3 and split into 2 branch
		#backward4-1
		for i in range(len(self.num_blocks_back)):
			blocks = []
			for j in range(self.num_blocks_back[i]):
				if j == 0:
					blocks.append(PostRes(self.nfb[i] * 2, self.nfb[i]))
				else:
					blocks.append(PostRes(self.nfb[i], self.nfb[i]))
			setattr(self, 'backward' + str(4-i), nn.Sequential(*blocks))

		self.drop = nn.Dropout3d(p=0.5, inplace=False)

		self.softmax = nn.Softmax(dim=1)#(NCDHW)

	def forward(self, layer1, layer2, layer3, layer4, layer5):
		# decoder
		up4 = self.deconv4(layer5)
		cat_4 = torch.cat((up4, layer4), 1)
		layer_4 = self.backward4(cat_4)

		up3 = self.deconv3(layer_4)
		cat_3 = torch.cat((up3, layer3), 1)
		layer_3 = self.backward3(cat_3)

		up2 = self.deconv2(layer_3)
		cat_2 = torch.cat((up2, layer2), 1)
		layer_2 = self.backward2(cat_2)

		up1 = self.deconv1(layer_2)
		cat_1 = torch.cat((up1, layer1), 1)
		layer_1 = self.backward1(cat_1)

		layer_1 = self.output(layer_1)
		layer_1 = self.softmax(layer_1)
		return layer_1

class TumorNet(nn.Module):
    def __init__(self):
        super(TumorNet, self).__init__()
        self. nff = [1, 8, 16, 32, 64, 128]#NumFeature_Forw[0-5]
        self.num_blocks_forw = [2, 2, 3, 3]#[2-5]
        # forward1
        self.forward1 = nn.Sequential(
            nn.Conv3d(self.nff[0], self.nff[1], kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.nff[1]),
            nn.ReLU(inplace=True),
            # nn.PReLU(),
            nn.Conv3d(self.nff[1], self.nff[1], kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.nff[1]),
            nn.ReLU(inplace=True)
            # nn.PReLU()
        )
        # forward2-5
        for i in range(len(self.num_blocks_forw)):  # 4
            blocks = []
            for j in range(self.num_blocks_forw[i]):  # {2,2,3,3}
                if j == 0:  # conv
                    ###plus source connection
                    blocks.append(PostRes(self.nff[i + 1], self.nff[i + 2]))
                else:
                    blocks.append(PostRes(self.nff[i + 2], self.nff[i + 2]))
            setattr(self, 'forward' + str(i + 2), nn.Sequential(*blocks))


        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        # downsamp1-4 by stride convolution
        # self.downsamp1 = nn.Conv3d(self.nff[1], self.nff[2], kernel_size=3, stride=2, padding=1)
        # self.downsamp2 = nn.Conv3d(self.nff[2], self.nff[3], kernel_size=3, stride=2, padding=1)
        # self.downsamp3 = nn.Conv3d(self.nff[3], self.nff[4], kernel_size=3, stride=2, padding=1)
        # self.downsamp4 = nn.Conv3d(self.nff[4], self.nff[5], kernel_size=3, stride=2, padding=1)

        self.decoder2 = Decoder2()
        self.drop = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        #encoder
        layer1 = self.forward1(input)
        down1 = self.maxpool(layer1)
        # down1 = self.downsamp1(layer1)

        layer2 = self.forward2(down1)
        down2 = self.maxpool(layer2)
        # down2 = self.downsamp2(layer2)

        layer3 = self.forward3(down2)
        down3 = self.maxpool(layer3)
        # down3 = self.downsamp3(layer3)

        layer4 = self.forward4(down3)
        down4 = self.maxpool(layer4)
        # down4 = self.downsamp4(layer4)

        layer5 = self.forward5(down4)
        # decoder
        branch2 = self.decoder2(layer1, layer2, layer3, layer4, layer5)
        return branch2

def main():
	net = TumorNet().cuda()#necessary for torchsummary, must to cuda
	from torchsummary import summary
	summary(net, input_size=(1,64,256,256))#must remove the number of N

	# input = torch.randn([1,1,64,256,256]).cuda()#(NCDHW)
	# output = net(input)
	# print(output.shape)

	# print('############net.named_parameters()#############')
	# for name, param in net.named_parameters():
	# 	print(name)

if __name__ == '__main__':
	main()