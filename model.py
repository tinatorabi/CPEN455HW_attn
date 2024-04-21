import torch.nn as nn
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.label_embedding = nn.Embedding(num_classes, nr_filters * 32 * 32)

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3), shift_output_down=True)
        self.ul_init = down_right_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,1), shift_output_right=True)

        # Setup the resnet blocks
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(nr_resnet, nr_filters, self.resnet_nonlinearity) for _ in range(3)])
        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity) for _ in range(3)])

        # Downsampling and upsampling streams
        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])
        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])
        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])

        self.nin_out = nin(nr_filters, nr_logistic_mix * 10)  # adjust number_mix based on actual application
        self.init_padding = None

    def forward(self, x, labels=None, sample=False):
        if self.init_padding is None:
            self.init_padding = Variable(torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]), requires_grad=False)
            self.init_padding = self.init_padding.cuda() if x.is_cuda else self.init_padding

        # Initial transformations
        u = self.u_init(torch.cat((x, self.init_padding), 1))
        ul = self.ul_init(torch.cat((x, self.init_padding), 1))

        # Embed and reshape labels if provided
        if labels is not None:
            label_emb = self.label_embedding(labels)
            label_emb = label_emb.view(x.shape[0], self.nr_filters, 32, 32)
            label_emb = F.interpolate(label_emb, size=(u.shape[2], u.shape[3]), mode='nearest')
            u += label_emb
            ul += label_emb

        # Process through the network
        u_list, ul_list = [u], [ul]
        for layer in self.up_layers:
            u, ul = layer(u_list[-1], ul_list[-1])
            u_list.append(u)
            ul_list.append(ul)

        # Downward pass
        u, ul = u_list.pop(), ul_list.pop()
        for layer in self.down_layers:
            u, ul = layer(u, ul, u_list.pop(), ul_list.pop())

        x_out = self.nin_out(F.elu(ul))
        return x_out
    
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if 'models' not in os.listdir():
            os.mkdir('models')
        torch.save(self.state_dict(), 'models/conditional_pixelcnn.pth')
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
    
    
