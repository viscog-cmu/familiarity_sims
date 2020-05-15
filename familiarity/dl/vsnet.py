
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import rescale


class DimRedConv(nn.Module):
    """
    implement a dimensionality-reduced convolution
    an important part of the Inception module, as well as computation
    in VsNet of Yu et. al (2019), Visual Cognition
    """

    def __init__(self, dim_in, dim_red, kernel_size, **kwargs):
        super().__init__()
        self.dim_reducer = nn.Conv2d(dim_in, dim_red, kernel_size=(1, 1))
        self.conv = nn.Conv2d(
            dim_red, dim_red, kernel_size=kernel_size, **kwargs)

    def forward(self, input):
        x1 = self.dim_reducer(input)
        x2 = self.conv(x1)
        return x2


class TiedDimRedConv(nn.Module):
    """
    implement a dimensionality-reduced convolution
    an important part of the Inception module, as well as computation
    in VsNet of Yu et. al (2019), Visual Cognition

    weights are tied to another layer DimRedConv layer with the same number in/out channel dim

    """

    def __init__(self, scale, tied_layer, **kwargs):
        super().__init__()
        self.scale = scale
        self.tied_layer = tied_layer
        # self.kwargs = kwargs

    def forward(self, input):
        weight = rescale(tied_layer.conv.weight, self.scale)
        x1 = F.conv2d(input, weight, tied_layer.conv.bias)
        x1 = self.dim_reducer(input)
        x2 = self.conv(x1)
        return x2


class ScaleInvConv2d(nn.Module):
    """
    scale-invariant convolution (Kanazawa, Sharma, Jacobs 2014)
    """

    def __init__(self, dim_in, dim_out, kernel_size, scales=[0.5, 1, 2], **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, **kwargs)
        self.scales = scales

    def forward(self, input):
        outputs = []
        for scale in self.scales:
            input_ = F.interpolate(input, scale_factor=scale, mode='bilinear')
            output = F.interpolate(self.conv(input_), size=input.shape[-2:])
            outputs.append(output)
        pooled_output = torch.max(torch.stack(outputs), 0)[0]
        return pooled_output


class MultiScaleConv(nn.Module):
    """
    implement weight-sharing across a multi scale convolution operation
    similar to the nception module, but with weight-sharing for size-invariance
    and fewer parameters
    """

    def __init__(self, dim_in, dim_out, kernel_sizes, **kwargs):
        super().__init__()
        for ii, kernel_size in enumerate(kernel_sizes):
            if ii == 0:
                exec(
                    f'self.scale{ii+1} = DimRedConv({dim_in}, {dim_red}, {kernel_size}, **{kwargs})')
            else:
                exec(
                    f'self.scale{ii+1} = TiedDimRedConv({dim_in}, {dim_red}, {kernel_size}, self.scale1, **{kwargs})')

    def forward(self, input):
        raise NotImplementedError()


class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class Concatenater(nn.Module):
    """
    Depthwise concatenates n_maps inputs with the same feature map size

    forward takes a tuple of n_maps
    """

    def __init__(self, n_maps):
        super().__init__()
        self.n_maps = n_maps

    def forward(self, x):
        assert len(x) == self.n_maps, print(
            f'{n_maps} inputs expected, got {len(x)}')
        return torch.cat(x, 1)


class BnReLU(nn.Module):
    """
    applies BatchNorm + ReLu with default parameters
    """

    def __init__(self, num_features, nd=2):
        super().__init__()
        if nd == 2:
            self.bn = nn.BatchNorm2d(num_features)
        else:
            self.bn = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x0):
        x1 = self.bn(x0)
        return self.relu(x1)


class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Vsnet(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __init__(self, n_classes=1000, initialize_weights=True):
        super().__init__()
        # define layers
        self.V1_s = nn.Conv2d(3, 24, kernel_size=(
            3, 3), stride=(2, 2), padding=(0, 0))
        self.V1_m = nn.Conv2d(3, 22, kernel_size=(
            5, 5), stride=(2, 2), padding=(1, 1))
        self.V1_l = nn.Conv2d(3, 18, kernel_size=(
            7, 7), stride=(2, 2), padding=(2, 2))
        self.V1_cat = Concatenater(3)
        self.V1_act = BnReLU(64, 2)
        self.V1_to_V4 = nn.Conv2d(64, 36, kernel_size=(1, 1))
        self.V2_s = DimRedConv(64, 30, kernel_size=(3, 3), padding=(1, 1))
        self.V2_m = DimRedConv(64, 28, kernel_size=(5, 5), padding=(2, 2))
        self.V2_l = DimRedConv(64, 24, kernel_size=(7, 7), padding=(3, 3))
        self.V2_cat = Concatenater(3)
        self.V2_act = BnReLU(82, 2)
        self.V2_pool = nn.MaxPool2d((3, 3), 2)
        self.V2_to_PIT = Identity()
        self.V4_collect = Concatenater(2)
        self.V4_s = DimRedConv(118, 60, kernel_size=(3, 3), padding=(1, 1))
        self.V4_m = DimRedConv(118, 50, kernel_size=(5, 5), padding=(2, 2))
        self.V4_cat = Concatenater(2)
        self.V4_act = BnReLU(110, 2)
        self.V4_to_AIT = Identity()
        self.PIT_collect = Concatenater(2)
        self.PIT_s = DimRedConv(192, 100, kernel_size=(3, 3), padding=(1, 1))
        self.PIT_m = DimRedConv(192, 100, kernel_size=(5, 5), padding=(2, 2))
        self.PIT_cat = Concatenater(2)
        self.PIT_act = BnReLU(200, 2)
        self.PIT_pool = nn.MaxPool2d((3, 3), 2)
        self.AIT_collect = Concatenater(2)
        self.AIT_s = DimRedConv(310, 91, kernel_size=(3, 3))
        self.AIT_m = DimRedConv(310, 91, kernel_size=(5, 5), padding=(1, 1))
        self.AIT_l = DimRedConv(310, 91, kernel_size=(7, 7), padding=(2, 2))
        self.AIT_cat = Concatenater(3)
        self.AIT_act = BnReLU(273, 2)
        self.AIT_pool = nn.MaxPool2d((5, 5), 4)
        self.fc1_flatten = Flatten()
        self.fc1 = nn.Linear(9828, 4096)
        self.fc1_act = BnReLU(4096, 1)
        self.fc2 = nn.Linear(4096, n_classes)
        self.prob = nn.Softmax(dim=1)

        if initialize_weights:
            self._initialize_weights()

    def forward(self, input):
        V1_out = self.V1_act(self.V1_cat(
            (self.V1_s(input), self.V1_m(input), self.V1_l(input))))
        V2_out = self.V2_act(self.V2_cat(
            (self.V2_s(V1_out), self.V2_m(V1_out), self.V2_l(V1_out))))
        V4_in = self.V2_pool(self.V4_collect((self.V1_to_V4(V1_out), V2_out)))
        V4_out = self.V4_act(self.V4_cat((self.V4_s(V4_in), self.V4_m(V4_in))))
        PIT_in = self.PIT_collect(
            (self.V2_to_PIT(self.V2_pool(V2_out)), V4_out))
        PIT_out = self.PIT_act(self.PIT_cat(
            (self.PIT_s(PIT_in), self.PIT_m(PIT_in))))
        AIT_in = self.PIT_pool(self.AIT_collect(
            (self.V4_to_AIT(V4_out), PIT_out)))
        AIT_out = self.AIT_act(self.AIT_cat(
            (self.AIT_s(AIT_in), self.AIT_m(AIT_in), self.AIT_l(AIT_in))))
        fc1_in = self.fc1_flatten(self.AIT_pool(AIT_out))
        fc1_out = self.fc1_act(self.fc1(fc1_in))
        fc2_out = self.fc2(fc1_out)
        prob = self.prob(fc2_out)
        return prob

    def forward_debug(self, input, return_layers, device='cpu'):
        a = {}
        a['V1_out'] = self.V1_act(self.V1_cat(
            (self.V1_s(input), self.V1_m(input), self.V1_l(input))))
        a['V2_out'] = self.V2_act(self.V2_cat(
            (self.V2_s(a['V1_out']), self.V2_m(a['V1_out']), self.V2_l(a['V1_out']))))
        a['V4_in'] = self.V2_pool(self.V4_collect(
            (self.V1_to_V4(a['V1_out']), a['V2_out'])))
        a['V4_out'] = self.V4_act(self.V4_cat(
            (self.V4_s(a['V4_in']), self.V4_m(V4_in))))
        a['PIT_in'] = self.PIT_collect(
            (self.V2_to_PIT(self.V2_pool(a['V2_out'])), a['V4_out']))
        a['PIT_out'] = self.PIT_act(self.PIT_cat(
            (self.PIT_s(a['PIT_in']), self.PIT_m(a['PIT_in']))))
        a['AIT_in'] = self.PIT_pool(self.AIT_collect(
            (self.V4_to_AIT(a['V4_out']), a['PIT_out'])))
        a['AIT_out'] = self.AIT_act(self.AIT_cat(
            (self.AIT_s(a['AIT_in']), self.AIT_m(a['AIT_in']), self.AIT_l(a['AIT_in']))))
        a['fc1_in'] = self.fc1_flatten(self.AIT_pool(a['AIT_out']))
        a['fc1_out'] = self.fc1(a['fc1_in'])
        a['fc2_out'] = self.fc2(a['fc1_out'])
        a['prob'] = self.prob(a['fc2_out'])

        out = {}
        for layer in return_layers:
            out[layer] = a.pop(layer).to(device)
        del a
        return out


def vsnet(weights_path=None, n_classes=1000, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    if weights_path:
        state_dict = torch.load(weights_path)
        if isinstance(state_dict, tuple):
            state_dict = state_dict[0]
            state_dict_ = OrderedDict()
            for key in state_dict.keys():
                state_dict_[key.replace('module.', '')] = state_dict[key]
            state_dict = state_dict_
        n_classes = len(state_dict['fc2.bias'])
    model = Vsnet(n_classes, initialize_weights=weights_path is None)
    if weights_path:
        model.load_state_dict(state_dict)
    model.meta = {'mean': [122.74494171142578, 114.94409942626953, 101.64177703857422],
                  'std': [1, 1, 1],
                  'imageSize': [224, 224, 3]}
    return model
