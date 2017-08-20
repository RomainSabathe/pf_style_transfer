import sys
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class VGG(nn.Module):
    def __init__(self, pool='max', conv_padding=1, conv_kernel_size=3,
                 pool_kernel_size=2, pool_stride=2):
        super(VGG, self).__init__()

        conv_layer = lambda in_channels, out_channels: nn.Conv2d(
            in_channels, out_channels, conv_kernel_size, padding=conv_padding)
        self.conv1_1 = conv_layer(3, 64)
        self.conv1_2 = conv_layer(64, 64)
        self.conv2_1 = conv_layer(64, 128)
        self.conv2_2 = conv_layer(128, 128)
        self.conv3_1 = conv_layer(128, 256)
        self.conv3_2 = conv_layer(256, 256)
        self.conv3_3 = conv_layer(256, 256)
        self.conv3_4 = conv_layer(256, 256)
        self.conv4_1 = conv_layer(256, 512)
        self.conv4_2 = conv_layer(512, 512)
        self.conv4_3 = conv_layer(512, 512)
        self.conv4_4 = conv_layer(512, 512)
        self.conv5_1 = conv_layer(512, 512)
        self.conv5_2 = conv_layer(512, 512)
        self.conv5_3 = conv_layer(512, 512)
        self.conv5_4 = conv_layer(512, 512)

        if pool == 'max':
            pool_layer = lambda: nn.MaxPool2d(pool_kernel_size,
                                             stride=pool_stride)
        elif pool == 'avg':
            pool_layer = lambda: nn.AvgPool2d(pool_kernel_size,
                                             stride=pool_stride)
        else:
            raise NotImplemented('This pool function is unknown.')
        self.pool1 = pool_layer()
        self.pool2 = pool_layer()
        self.pool3 = pool_layer()
        self.pool4 = pool_layer()
        self.pool5 = pool_layer()

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        flat = input.view(b, c, h*w)  # flatten out the feature maps.
        gram = torch.bmm(flat, flat.transpose(1, 2))
        gram.div_(h*w)
        return gram


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        main_loss = nn.MSELoss()
        gram = GramMatrix()
        input_gram = gram(input)
        return main_loss(input_gram , target)

# Pre/postprocessing
img_size = 512

preprocessing_flow = transforms.Compose([
    transforms.Scale(img_size),
    transforms.ToTensor(),
    # convert to BGR
    transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
    # substact image mean.
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                         std=[1, 1, 1]),
    transforms.Lambda(lambda x: x.mul_(255))])

postprocessing_flow = transforms.Compose([
    transforms.Lambda(lambda x: x.mul_(1./255)),
    # add image mean.
    transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                         std=[1, 1, 1]),
    # convert to RGB
    transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])])

def tensor_to_pil_image(tensor):
    """Takes a tensor that hasn't been yet postprocessed and transfoms it
    into a valid PIL image."""
    tensor = postprocessing_flow(tensor)
    # Clip results in the range [0, 1].
    tensor[tensor>1] = 1
    tensor[tensor<0] = 0
    return transforms.ToPILImage()(tensor)


# Load network.
vgg = VGG()
vgg.load_state_dict(torch.load('style_transfer/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()


def main(style_img, target_img):
    # CONVENTION: if a variable has a `t_` prefix, it means it's a tensor.

    # Loading and preprocessing the images.
    images = [style_img, target_img]
    t_images = [preprocessing_flow(img) for img in images]
    t_images = [Variable(t_img.unsqueeze(0)) for t_img in t_images]
    if torch.cuda.is_available():
        t_images = [t_img.cuda() for t_img in t_images]
    t_style_image, t_target_image = t_images

    t_output_img = Variable(t_target_image.data.clone(), requires_grad=True)

    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_functions = [GramMSELoss()] * len(style_layers) + \
                     [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_functions = [loss_fn.cuda() for loss_fn in loss_functions]

    # Empirical weights settings.
    style_weights = [1e3/x**2 for x in [64, 128, 256, 512, 512]]
    content_weights = [1e0 for _ in content_layers]
    weights = style_weights + content_weights

    # Compute optimization targets.
    style_features = vgg(t_style_image, style_layers)
    style_targets = [GramMatrix()(features) for features in style_features]
    content_features = vgg(t_target_image, content_layers)
    content_targets = [features.detach() for features in content_features]
    targets = style_targets + content_targets

    # Run style transfer.
    max_iter = 10
    show_iter = 1
    optimizer = optim.LBFGS([t_output_img])

    for n_iter in range(max_iter):
        def closure():
            optimizer.zero_grad()
            endpoints = vgg(t_output_img, loss_layers)
            layer_losses = [weight * loss_function(endpoint, target) for \
                            (weight, target, loss_function, endpoint) in \
                            zip(weights, targets, loss_functions, endpoints)]
            loss = sum(layer_losses)
            loss.backward()

            if n_iter % show_iter == 0:
                logging.debug('Iteration {}, loss: {:.4f}'.format(
                    n_iter, loss.data[0]))

            return loss
        optimizer.step(closure)

    return tensor_to_pil_image(t_output_img.data[0].cpu().squeeze())


if __name__ == '__main__':
    from PIL import Image
    style_image = Image.open('tmp/style_img.jpg')
    target_image = Image.open('tmp/target_img.jpg')
    main(style_image, target_image)
