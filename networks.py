import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class Proj(nn.Module):
    """
    a two-layer MLP for classification
    """

    def __init__(self, in_dim, out_dim):
        super(Proj, self).__init__()
        self.bottleneck = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, nn.ReLU())
        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = [x]
        for module in self.main.children():
            # print("cls input x: ", x.size())
            x = module(x)
            out.append(x)
        
        return out

    def get_parameters(self):
        parameter_list = [{"params": self.main.parameters(), "lr_mult": 1, 'decay_mult': 2}]

        return parameter_list
    

class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """

    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))
        

    def forward(self, x):
        # print('cls input x: ', x.size())
        x = x.view(x.size(0), -1)
        out = [x]
        for module in self.main.children():
            # print("cls input x: ", x.size())
            x = module(x)
            out.append(x)
        
        return out

    def get_parameters(self):
        parameter_list = [{"params": self.main.parameters(), "lr_mult": 1, 'decay_mult': 2}]

        return parameter_list

class CLS_ML(nn.Module):
    """
    a two-layer MLP for classification
    """

    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS_ML, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Sigmoid())
        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = [x]
        for module in self.main.children():
            # print("cls input x: ", x.size())
            x = module(x)
            out.append(x)
        # print('out[3]: ', out[3].type(), out[3])
        return out

    def get_parameters(self):
        parameter_list = [{"params": self.main.parameters(), "lr_mult": 1, 'decay_mult': 2}]

        return parameter_list
    
class ResNet18Fc(nn.Module):
    def __init__(self):
        super(ResNet18Fc, self).__init__()
        model_resnet18 = models.resnet18(pretrained = True)#weights = ResNet18_Weights.DEFAULT)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # # print('!!!!x shape is', x.size())
        
        # y = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class ResNet18Fc_modified(nn.Module):
    def __init__(self):
        super(ResNet18Fc_modified, self).__init__()
        model_resnet18 = models.resnet18(pretrained = True)#weights = ResNet18_Weights.DEFAULT)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # print('!!!!x shape is', x.size())
        
        y = x.view(x.size(0), -1)
        return x, y

    def output_num(self):
        return self.__in_features

class ResNet34Fc(nn.Module):
    def __init__(self):
        super(ResNet34Fc, self).__init__()
        model_resnet34 = models.resnet34(pretrained=True)
        # print(model_resnet34)
        self.__in_features = model_resnet34.fc.in_features
        model_resnet34.fc = nn.Linear(self.__in_features, 512)
        self.conv1 = model_resnet34.conv1
        self.bn1 = model_resnet34.bn1
        self.relu = model_resnet34.relu
        self.maxpool = model_resnet34.maxpool
        self.layer1 = model_resnet34.layer1
        self.layer2 = model_resnet34.layer2
        self.layer3 = model_resnet34.layer3
        self.layer4 = model_resnet34.layer4
        self.avgpool = model_resnet34.avgpool
        self.fc = model_resnet34.fc
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # print(x.size())
        # x = self.fc(x)
        return x

    def output_num(self):
        return self.__in_features
    
class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        # print(model_resnet50)
        self.__in_features = model_resnet50.fc.in_features
        model_resnet50.fc = nn.Linear(self.__in_features, 512)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.fc = model_resnet50.fc
        
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # print(x.size())
        # x = self.fc(x)
        return x

    def output_num(self):
        return self.__in_features

# convnet without the last layer
class AlexNetFc(nn.Module):
    def __init__(self):
        super(AlexNetFc, self).__init__()
        # model_alexnet = models.alexnet(pretrained=True)
        model_alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()

        #for i in range(6):
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        # The following lines are added to convert the output size

        self.fc1 = nn.Linear(4096,1024)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.__in_features = model_alexnet.classifier[6].in_features



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        # Add one more layer to convert the features from 4096 to 1024, since the size 4096 will lead to 256G VRAM
        # this made the model architecture slightly different from the original one.
        # Should be noted in the text
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #print(x.size())
        x = x.view((x.size(0),256, 1, 1))
        # print(x.size())

        return x

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        parameter_list = [{"params": self.features.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                          {"params": self.classifier.parameters(), "lr_mult": 1, 'decay_mult': 2}]

        return parameter_list

class AlexNetFc_modified(nn.Module):
    def __init__(self):
        super(AlexNetFc_modified, self).__init__()
        # model_alexnet = models.alexnet(pretrained=True)
        model_alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()

        #for i in range(6):
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        # The following lines are added to convert the output size

        self.fc1 = nn.Linear(4096,1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.__in_features = model_alexnet.classifier[6].in_features



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        
        # Add one more layer to convert the features from 4096 to 1024, since the size 4096 will lead to 256G VRAM
        # this made the model architecture slightly different from the original one.
        # Should be noted in the text
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #print(x.size())
        x = x.view((x.size(0),512, 1, 1))
        y = x.view(x.size(0), -1)

        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        parameter_list = [{"params": self.features.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                          {"params": self.classifier.parameters(), "lr_mult": 1, 'decay_mult': 2}]

        return parameter_list


class Discriminator_network(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(Discriminator_network, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        #self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        #x_ = self.grl(x)
        y = self.main(x)
        return y
    
    def get_parameters(self):
        
        parameter_list = [{"params":self.main.parameters(), "lr_mult":1, 'decay_mult':2}]

        return parameter_list


class Alexnet_Discriminator_network(nn.Module):

    def __init__(self, in_feature):
        super(Alexnet_Discriminator_network, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        
        x_ = self.grl(x)
        y = self.main(x_)

        return y
    
    def get_parameters(self):
        
        parameter_list = [{"params":self.main.parameters(), "lr_mult":1, 'decay_mult':2}]

        return parameter_list


    
class GradientReverseLayer(torch.autograd.Function):
    """
    usage:(can't be used in nn.Sequential, not a subclass of nn.Module)::
        x = Variable(torch.ones(1, 2), requires_grad=True)
        grl = GradientReverseLayer.apply
        y = grl(0.5, x)
        y.backward(torch.ones_like(y))
        print(x.grad)
    """
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        # this is necessary. if we just return ``input``, ``backward`` will not be called sometimes
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs


class GradientReverseModule(nn.Module):
    """
    wrap GradientReverseLayer to be a nn.Module so that it can be used in ``nn.Sequential``
    usage::
        grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))
        x = Variable(torch.ones(1), requires_grad=True)
        ans = []
        for _ in range(10000):
            x.grad = None
            y = grl(x)
            y.backward()
            ans.append(variable_to_numpy(x.grad))
        plt.plot(list(range(10000)), ans)
        plt.show() # you can see gradient change from 0 to -1
    """
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.register_buffer('global_step', torch.zeros(1))
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x):
        self.coeff = self.scheduler(self.global_step.item())
        if self.training:
            self.global_step += 1.0
        return self.grl(self.coeff, x)    


def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    '''
    change as initial_lr * (1 + gamma * min(1.0, iter / max_iter) ) ** (- power)
    as known as inv learning rate sheduler in caffe,
    see https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
    the default gamma and power come from <Domain-Adversarial Training of Neural Networks>
    code to see how it changes(decays to %20 at %10 * max_iter under default arg)::
        from matplotlib import pyplot as plt
        ys = [inverseDecaySheduler(x, 1e-3) for x in range(10000)]
        xs = [x for x in range(10000)]
        plt.plot(xs, ys)
        plt.show()
    '''
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))


def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    '''
    change gradually from A to B, according to the formula (from <Importance Weighted Adversarial Nets for Partial Domain Adaptation>)
    A + (2.0 / (1 + exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    =code to see how it changes(almost reaches B at %40 * max_iter under default arg)::
        from matplotlib import pyplot as plt
        ys = [aToBSheduler(x, 1, 3) for x in range(10000)]
        xs = [x for x in range(10000)]
        plt.plot(xs, ys)
        plt.show()
    '''
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)

network_dict = {"AlexNet": AlexNetFc}
