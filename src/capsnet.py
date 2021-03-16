import torch
import torch.nn as nn
import torch.nn.functional as F

from primarycaps import PrimaryCaps
from digitcaps import DigitCaps
from decoder import DenseDecoder, DeconvDecoder

class CapsuleNetwork(nn.Module):
	def __init__(self, decoder_type="deconv"):
		super(CapsuleNetwork, self).__init__()
		self.conv_layer = nn.Conv2d(1, 256, kernel_size=9, 
									stride=1, padding=0)
		self.primary_capsule = PrimaryCaps()
		self.digit_capsule = DigitCaps()
		if(decoder_type == "deconv"):
			self.decoder = DeconvDecoder()
		else:
			self.decoder = DenseDecoder()

	def forward(self, images):
		'''
		param images: MNIST input data
		return: output of DigitCaps, reconstructed images and class scores
		'''

		x = self.primary_capsule(self.conv_layer(images))
		digit_caps_out = self.digit_capsule(x).squeeze().transpose(0, 1)
		reconstructions, y = self.decoder(digit_caps_out)

		return digit_caps_out, reconstructions, y

