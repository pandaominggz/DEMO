import torch
import torch.nn as nn
import torch.nn.functional as F
from patch_match import PatchMatch


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.patch_match = PatchMatch(3)
        self.max_disp = 160
        self.conv0 = nn.Conv2d(3, 32, 5, 1, 2)
        self.bn0 = nn.BatchNorm2d(32)
        self.res_block = self.res_layers(BasicBlock, 32, 32, 8, stride=1)
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

    def res_layers(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for step in strides:
            layers.append(block(in_planes, planes, step))
        return nn.Sequential(*layers)

    def generate_disparity_samples(self, left_input, right_input, min_disparity,
                                   max_disparity, sample_count=10, sampler_type="patch_match"):
        """
        Description:    Generates "sample_count" number of disparity samples from the
                                                            search range (min_disparity, max_disparity)
                        Samples are generated either uniformly from the search range
                                                            or are generated using PatchMatch.

        Args:
            :left_input: Left Image features.
            :right_input: Right Image features.
            :min_disparity: LowerBound of the disaprity search range.
            :max_disparity: UpperBound of the disparity search range.
            :sample_count (default:10): Number of samples to be generated from the input search range.
            :sampler_type (default:"patch_match"): samples are generated either using
                                                                    "patch_match" or "uniform" sampler.
        Returns:
            :disparity_samples:
        """
        if sampler_type is "patch_match":
            disparity_samples = self.patch_match(left_input, right_input, min_disparity,
                                                 max_disparity, sample_count, 3)
        else:
            disparity_samples = self.uniform_sampler(min_disparity, max_disparity, sample_count)

        disparity_samples = torch.cat((torch.floor(min_disparity), disparity_samples, torch.ceil(max_disparity)),
                                      dim=1)
        return disparity_samples

    def generate_search_range(self, left_input, sample_count, stage, input_min_disparity=None,
                              input_max_disparity=None):
        """
        Description:    Generates the disparity search range depending upon the stage it is called.
                    If stage is "pre" (Pre-PatchMatch and Pre-ConfidenceRangePredictor), the search range is
                    the entire disparity search range.
                    If stage is "post" (Post-ConfidenceRangePredictor), then the ConfidenceRangePredictor search range
                    is adjusted for maximum efficiency.
        Args:
            :left_input: Left Image Features
            :sample_count: number of samples to be generated from the search range. Used to adjust the search range.
            :stage: "pre"(Pre-PatchMatch) or "post"(Post-ConfidenceRangePredictor)
            :input_min_disparity (default:None): ConfidenceRangePredictor disparity lowerbound (for stage=="post")
            :input_max_disparity (default:None): ConfidenceRangePredictor disparity upperbound (for stage=="post")

        Returns:
            :min_disparity: Lower bound of disparity search range
            :max_disparity: Upper bound of disaprity search range.
        """

        device = left_input.get_device()
        if stage is "pre":
            min_disparity = torch.zeros((left_input.size()[0], 1, left_input.size()[2], left_input.size()[3]),
                                        device=device)
            max_disparity = torch.zeros((left_input.size()[0], 1, left_input.size()[2], left_input.size()[3]),
                                        device=device) + self.max_disp

        else:
            min_disparity1 = torch.min(input_min_disparity, input_max_disparity)
            max_disparity1 = torch.max(input_min_disparity, input_max_disparity)
            min_disparity = torch.clamp(min_disparity1 - torch.clamp((
                    sample_count - max_disparity1 + min_disparity1), min=0) / 2.0, min=0, max=self.max_disp)
            max_disparity = torch.clamp(max_disparity1 + torch.clamp(
                sample_count - max_disparity1 + min_disparity, min=0), min=0, max=self.max_disp)

        return min_disparity, max_disparity

    def forward(self, imgL, imgR):
        imgL = F.relu(self.bn0(self.conv0(imgL)))
        imgR = F.relu(self.bn0(self.conv0(imgR)))

        imgL_block = self.res_block(imgL)
        imgR_block = self.res_block(imgR)

        imgL = self.conv1(imgL_block)
        imgR = self.conv1(imgR_block)

        min_disparity, max_disparity = self.generate_search_range(imgL, 10, stage="pre")
        disparity_samples = self.generate_disparity_samples(imgL, imgR, min_disparity, max_disparity, 10,
                                                            sampler_type="patch_match")

        # disparity_samples = torch.mean(disparity_samples, 1)
        return disparity_samples
