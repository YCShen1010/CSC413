import torch
import torch.nn as nn


class FSRCNN(torch.nn.Module):
    def __init__(self, num_channels, upscale_factor, d=56, s=12, m=4):
        ''' Create the FSRCNN with five layers with given parameters d, s, and m '''

        super(FSRCNN, self).__init__()

        # extract the reature by a 5*5 kernel
        self.feature_extraction = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
                                        nn.PReLU())

        # do the shrinking, mapping, and expanding
        # reduce the LR feature dimension d to s by 1*1 kernel
        self.shrink_expand_part = [nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, padding=0), nn.PReLU(s)]

        # do the mapping according to the given m
        for x in range(m):
            self.shrink_expand_part.extend([nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1), nn.PReLU(s)])

        # expand s back to the HR feature dimension d by 1*1 kernel
        self.shrink_expand_part.extend([nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1,padding=0), nn.PReLU(d)])
        self.shrink_expand_part = torch.nn.Sequential(*self.shrink_expand_part)

        # deconvolution by 9*9 kernel to rebuild the RH images
        self.deconvolution = nn.ConvTranspose2d(in_channels=d, out_channels=num_channels, kernel_size=9, stride=upscale_factor, padding=4, output_padding=upscale_factor-1)


    def forward(self, x):
        ''' By the FSRCNN, forward is used to apply to all three main layers '''
        out = self.feature_extraction(x)
        out = self.shrink_expand_part(out)
        out = self.deconvolution(out)
        return out

    def weight_init(self, mean, std):
        ''' give the init for weight and bias'''

        # torch.nn.init.normal(tensor, mean, std)
        # for feature extraction layer
        for j in self.feature_extraction:
            if isinstance(j, nn.Conv2d):
                nn.init.normal_(j.weight, mean=0, std=0.0378)
                nn.init.constant_(j.bias, 0)

        # for shrinking and expanding layers
        for k in self.shrink_expand_part:
            if isinstance(k, nn.Conv2d):
                nn.init.normal_(k.weight, mean=0, std=0.1179)
                nn.init.constant_(k.bias, 0)

        # for deconvolution layers
        nn.init.normal_(self.deconvolution.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.deconvolution.bias, 0)
