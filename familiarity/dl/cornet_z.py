from collections import OrderedDict
from torch import nn
from torch import load

from familiarity.dl.vsnet import ScaleInvConv2d
from familiarity.dl.lc import Conv2dLocal

class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x

class EConv2d(nn.Conv2d):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp_(min=0)
        return super().forward(inputs)

class IConv2d(nn.Conv2d):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp_(max=0)
        return super.forward(inputs)


class ELinear(nn.Linear):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp(min=0)
        return super().forward(inputs)

class ILinear(nn.Linear):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp_(max=0)
        super.forward(inputs)

class AdaptiveLinear(nn.Linear):
    """
        a linear layer that automatically determines the number of inputs on initial forward pass
        must remain the same for all future forward passes

        ** note that you must set the optimizer AFTER adapting this layer **
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapted = False
        self.flatten = Flatten()
    def forward(self, inputs):
        if not self.adapted:
            inputs = self.flatten(inputs)
            ndim_in = inputs.shape[1]
            new_units = torch.nn.Linear(ndim_in, self.out_features)
            device = self._parameters['weight'].data.get_device()
            self._parameters['weight'].data = new_units._parameters['weight'].data.to(device)
            self.adapted = True
        return super().forward(inputs)

class AdaptiveELinear(AdaptiveLinear):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp(min=0)
        return super().forward(inputs)

# class EIConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
#         self.conv = EConv2d(in_channels, out_channels, kernel_size, *args, **kwargs)
#         self.e2i = nn.AdaptiveELinear(10, 1)
#         self.i2e

class CORblock_Z(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, conv_scales=None, locally_connected=False,
    in_height=None,
    in_width=None):
        super().__init__()
        if conv_scales is not None:
            if locally_connected:
                raise NotImplementedError()
            self.conv = ScaleInvConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size//2,
                                    scales=conv_scales)
        else:
            if locally_connected:
                self.conv = Conv2dLocal(in_height, in_width, in_channels, out_channels,
                                    kernel_size=kernel_size, padding=kernel_size // 2,
                                    stride=stride)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x

class CORnet_Z(nn.Sequential):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        # weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_debug(self, x0, return_layers, device='cpu'):
        acts = {}
        acts['x0'] = x0
        acts['x1'] = self.V1(x0)
        acts['x2'] = self.V2(acts['x1'])
        acts['x3'] = self.V4(acts['x2'])
        acts['x4'] = self.IT(acts['x3'])
        acts['x5'] = self.decoder(acts['x4'])
        acts['x6'] = nn.functional.softmax(acts['x5'], 1)

        out = {}
        for layer in return_layers:
            out['x{}'.format(layer)] = acts.pop('x{}'.format(layer)).to(device)
        del acts
        return out

def cornet_z(weights_path=None, n_classes=1000,
                half_filters_at_layer=None,
                conv_scales=None):
    n_filts = dict(V1=64, V2=128, V4=256, IT=512)
    if half_filters_at_layer:
        assert half_filters_at_layer in ['V1', 'V2', 'V4', 'IT', 'ALL'], f'invalid layer {half_filters_at_layer} specified for filter halfing'
        if half_filters_at_layer == 'ALL':
            n_filts = {key: int(val/2) for key, val in n_filts.items()}
        else:
            n_filts[half_filters_at_layer] = int(n_filts[half_filters_at_layer]/2)
    if weights_path:
        state_dict = load(weights_path)
        if type(state_dict) is tuple:
            state_dict = state_dict[0]
        state_dict_ = OrderedDict()
        for key, val in state_dict.items():
            state_dict_[key.replace('module.', '')] = val
        n_classes = len(state_dict_['decoder.linear.bias'])
    model = CORnet_Z(OrderedDict([
        ('V1', CORblock_Z(3, n_filts['V1'], kernel_size=7, stride=2, conv_scales=conv_scales)),
        ('V2', CORblock_Z(n_filts['V1'], n_filts['V2'], conv_scales=conv_scales)),
        ('V4', CORblock_Z(n_filts['V2'], n_filts['V4'], conv_scales=conv_scales)),
        ('IT', CORblock_Z(n_filts['V4'], n_filts['IT'], conv_scales=conv_scales)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(n_filts['IT'], n_classes)),
            ('output', Identity())
        ])))]))
    if weights_path:
        model.load_state_dict(state_dict_)
    return model




class BranchedCORnet_Z(nn.Module):

    def __init__(self, weights_path=None, n_classes=1000,
                branch_point=None,
                n_classes_branch1=None,
                conv_scales=None):
        super().__init__()
        n_filts = dict(V1=64, V2=128, V4=256, IT=512)
        if weights_path:
            state_dict = load(weights_path)
            if type(state_dict) is tuple:
                state_dict = state_dict[0]
            state_dict_ = OrderedDict()
            for key, val in state_dict.items():
                state_dict_[key.replace('module.', '')] = val
            if branch_point is not None:
                n_classes_branch1 = len(state_dict_['decoder_1.linear.bias'])
                n_classes = n_classes_branch1 + len(state_dict_['decoder_2.linear.bias'])
            else:
                n_classes = len(state_dict_['decoder.linear.bias'])
        self.branch_point = branch_point
        if branch_point == 'V1':
            self.V1_1 = CORblock_Z(3, n_filts['V1']//2, kernel_size=7, stride=2, conv_scales=conv_scales)
            self.V1_2 = CORblock_Z(3, n_filts['V1']//2, kernel_size=7, stride=2, conv_scales=conv_scales)
        else:
            self.V1 = CORblock_Z(3, n_filts['V1'], kernel_size=7, stride=2, conv_scales=conv_scales)
        if branch_point in ['V1', 'V2']:
            in_filts = n_filts['V1'] if branch_point == 'V2' else n_filts['V1']//2
            self.V2_1 = CORblock_Z(in_filts, n_filts['V2']//2, conv_scales=conv_scales)
            self.V2_2 = CORblock_Z(in_filts, n_filts['V2']//2, conv_scales=conv_scales)
        else:
            self.V2 = CORblock_Z(n_filts['V1'], n_filts['V2'], conv_scales=conv_scales)
        if branch_point in ['V1', 'V2', 'V4']:
            in_filts = n_filts['V2'] if branch_point == 'V4' else n_filts['V2']//2
            self.V4_1 = CORblock_Z(in_filts, n_filts['V4']//2, conv_scales=conv_scales)
            self.V4_2 = CORblock_Z(in_filts, n_filts['V4']//2, conv_scales=conv_scales)
        else:
            self.V4 = CORblock_Z(n_filts['V2'], n_filts['V4'], conv_scales=conv_scales)
        if branch_point in ['V1', 'V2', 'V4', 'IT']:
            in_filts = n_filts['V4'] if branch_point == 'IT' else n_filts['V4']//2
            self.IT_1 = CORblock_Z(in_filts, n_filts['IT']//2, conv_scales=conv_scales)
            self.IT_2 = CORblock_Z(in_filts, n_filts['IT']//2, conv_scales=conv_scales)
        else:
            self.IT = CORblock_Z(n_filts['V4'], n_filts['IT'], conv_scales=conv_scales)
        if branch_point in ['V1', 'V2', 'V4', 'IT', 'decoder']:
            in_filts = n_filts['IT'] if branch_point == 'decoder' else n_filts['IT']//2
            self.decoder_1 = nn.Sequential(OrderedDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', Flatten()),
                ('linear', nn.Linear(in_filts, n_classes_branch1)),
                ('output', Identity())]))
            self.decoder_2 = nn.Sequential(OrderedDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', Flatten()),
                ('linear', nn.Linear(in_filts, n_classes-n_classes_branch1)),
                ('output', Identity())]))
        else:
            self.decoder = nn.Sequential(OrderedDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', Flatten()),
                ('linear', nn.Linear(n_filts['IT'], n_classes)),
                ('output', Identity())]))

        if weights_path:
            self.load_state_dict(state_dict_)
        else:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, inputs):
        if self.branch_point is None:
            V1 = self.V1(inputs)
            V2 = self.V2(V1)
            V4 = self.V4(V2)
            IT = self.IT(V4)
            outputs = self.decoder(IT)
            return outputs
        if self.branch_point == 'V1':
            V1_1 = self.V1_1(inputs)
            V1_2 = self.V1_2(inputs)
            V2_1 = self.V2_1(V1_1)
            V2_2 = self.V2_2(V1_2)
            V4_1 = self.V4_1(V2_1)
            V4_2 = self.V4_2(V2_2)
            IT_1 = self.IT_1(V4_1)
            IT_2 = self.IT_2(V4_2)
            outputs_1 = self.decoder_1(IT_1)
            outputs_2 = self.decoder_2(IT_2)
            return torch.cat((outputs_1, outputs_2), dim=1)
        elif self.branch_point == 'V2':
            V1 = self.V1(inputs)
            V2_1 = self.V2_1(V1)
            V2_2 = self.V2_2(V1)
            V4_1 = self.V4_1(V2_1)
            V4_2 = self.V4_2(V2_2)
            IT_1 = self.IT_1(V4_1)
            IT_2 = self.IT_2(V4_2)
            outputs_1 = self.decoder_1(IT_1)
            outputs_2 = self.decoder_2(IT_2)
            return torch.cat((outputs_1, outputs_2), dim=1)
        elif self.branch_point == 'V4':
            V1 = self.V1(inputs)
            V2 = self.V2(V1)
            V4_1 = self.V4_1(V2)
            V4_2 = self.V4_2(V2)
            IT_1 = self.IT_1(V4_1)
            IT_2 = self.IT_2(V4_2)
            outputs_1 = self.decoder_1(IT_1)
            outputs_2 = self.decoder_2(IT_2)
            return torch.cat((outputs_1, outputs_2), dim=1)
        elif self.branch_point == 'IT':
            V1 = self.V1(inputs)
            V2 = self.V2(V1)
            V4 = self.V4(V2)
            IT_1 = self.IT_1(V4)
            IT_2 = self.IT_2(V4)
            outputs_1 = self.decoder_1(IT_1)
            outputs_2 = self.decoder_2(IT_2)
            return torch.cat((outputs_1, outputs_2), dim=1)
        elif self.branch_point == 'decoder':
            V1 = self.V1(inputs)
            V2 = self.V2(V1)
            V4 = self.V4(V2)
            IT = self.IT(V4)
            outputs_1 = self.decoder_1(IT)
            outputs_2 = self.decoder_2(IT)
            return torch.cat((outputs_1, outputs_2), dim=1)

    def forward_debug(self, inputs, return_layers, device='cpu'):
        acts=OrderedDict()
        acts['x0'] = inputs
        if self.branch_point is None:
            acts['x1'] = self.V1(inputs)
            acts['x2'] = self.V2(acts['x1'])
            acts['x3'] = self.V4(acts['x2'])
            acts['x4'] = self.IT(acts['x3'])
            acts['x5'] = self.decoder(acts['x4'])
        elif self.branch_point == 'decoder':
            acts['x1'] = self.V1(inputs)
            acts['x2'] = self.V2(acts['x1'])
            acts['x3'] = self.V4(acts['x2'])
            acts['x4'] = self.IT(acts['x3'])
            outputs_1 = self.decoder_1(acts['x4'])
            outputs_2 = self.decoder_2(acts['x4'])
            acts['x5'] = torch.cat((outputs_1, outputs_2), dim=1)
        elif self.branch_point == 'IT':
            acts['x1'] = self.V1(inputs)
            acts['x2'] = self.V2(acts['x1'])
            acts['x3'] = self.V4(acts['x2'])
            IT_1 = self.IT_1(acts['x3'])
            IT_2 = self.IT_2(acts['x3'])
            acts['x4'] = torch.cat((IT_1, IT_2), dim=1)
            outputs_1 = self.decoder_1(IT_1)
            outputs_2 = self.decoder_2(IT_2)
            acts['x5'] = torch.cat((outputs_1, outputs_2), dim=1)
        elif self.branch_point == 'V4':
            acts['x1'] = self.V1(inputs)
            acts['x2'] = self.V2(acts['x1'])
            V4_1 = self.V4_1(acts['x2'])
            V4_2 = self.V4_2(acts['x2'])
            acts['x3'] = torch.cat((V4_1, V4_2), dim=1)
            IT_1 = self.IT_1(V4_1)
            IT_2 = self.IT_2(V4_2)
            acts['x4'] = torch.cat((IT_1, IT_2), dim=1)
            outputs_1 = self.decoder_1(IT_1)
            outputs_2 = self.decoder_2(IT_2)
            acts['x5'] = torch.cat((outputs_1, outputs_2), dim=1)
        elif self.branch_point == 'V2':
            acts['x1'] = self.V1(inputs)
            V2_1 = self.V2_1(acts['x1'])
            V2_2 = self.V2_2(acts['x1'])
            acts['x2'] = torch.cat((V2_1, V2_2), dim=1)
            V4_1 = self.V4_1(V2_1)
            V4_2 = self.V4_2(V2_2)
            acts['x3'] = torch.cat((V4_1, V4_2), dim=1)
            IT_1 = self.IT_1(V4_1)
            IT_2 = self.IT_2(V4_2)
            acts['x4'] = torch.cat((IT_1, IT_2), dim=1)
            outputs_1 = self.decoder_1(IT_1)
            outputs_2 = self.decoder_2(IT_2)
            acts['x5'] = torch.cat((outputs_1, outputs_2), dim=1)
        if self.branch_point == 'V1':
            V1_1 = self.V1_1(inputs)
            V1_2 = self.V1_2(inputs)
            acts['x1'] = torch.cat((V1_1, V1_2), dim=1)
            V2_1 = self.V2_1(V1_1)
            V2_2 = self.V2_2(V1_2)
            acts['x2'] = torch.cat((V2_1, V2_2), dim=1)
            V4_1 = self.V4_1(V2_1)
            V4_2 = self.V4_2(V2_2)
            acts['x3'] = torch.cat((V4_1, V4_2), dim=1)
            IT_1 = self.IT_1(V4_1)
            IT_2 = self.IT_2(V4_2)
            acts['x4'] = torch.cat((IT_1, IT_2), dim=1)
            outputs_1 = self.decoder_1(IT_1)
            outputs_2 = self.decoder_2(IT_2)
            acts['x5'] = torch.cat((outputs_1, outputs_2), dim=1)
        acts['x6'] = nn.functional.softmax(acts['x5'], 1)

        out = {}
        for layer in return_layers:
            out['x{}'.format(layer)] = acts.pop('x{}'.format(layer)).to(device)
        del acts
        return out


# def CORnet_Z(n_classes=1000):
#     model = nn.Sequential(OrderedDict([
#         ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
#         ('V2', CORblock_Z(64, 128)),
#         ('V4', CORblock_Z(128, 256)),
#         ('IT', CORblock_Z(256, 512)),
#         ('decoder', nn.Sequential(OrderedDict([
#             ('avgpool', nn.AdaptiveAvgPool2d(1)),
#             ('flatten', Flatten()),
#             ('linear', nn.Linear(512, n_classes)),
#             ('output', Identity())
#         ])))
#     ]))
#
#     # weight initialization
#     for m in model.modules():
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#
#     return model
