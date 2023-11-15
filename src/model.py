import copy
import torch
import torchvision



class EquiMod(torch.nn.Module):
    def __init__(self, resnet_type='resnet50', z_dim=2048, y_dim=2048, p_dim=17, cifar10=False, proj_head_eq_layers=None, proj_head_t_layers=None, predictor_eq_layers=None):
        super().__init__()
        resnet_dict = {
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152
        }

        self.resnet = resnet_dict[resnet_type](pretrained=False, zero_init_residual=True)
        self.h_dim = self.resnet.fc.in_features
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.p_dim = p_dim

        self.resnet.fc = torch.nn.Identity()
        if cifar10:
            self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.resnet.maxpool = torch.nn.Identity()

        self.proj_head_inv = torch.nn.Sequential(
            torch.nn.Linear(self.h_dim, 2048, bias=False),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048, bias=False),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.z_dim, bias=False),
            torch.nn.BatchNorm1d(self.z_dim, affine=False)
        )

        self.proj_head_eq = None
        if proj_head_eq_layers != "none":
            layers = []
            last = self.h_dim
            for n in map(lambda x: int(x), proj_head_eq_layers.split('-')[:-1]):
                layers.append(torch.nn.Linear(last, n, bias=False))
                layers.append(torch.nn.BatchNorm1d(n))
                layers.append(torch.nn.ReLU())
                last = n

            layers.append(torch.nn.Linear(last, self.y_dim, bias=False))
            layers.append(torch.nn.BatchNorm1d(self.y_dim, affine=False))

            self.proj_head_eq = torch.nn.Sequential(*layers)
        else:
            self.y_dim = self.h_dim

        self.proj_head_t = None
        if proj_head_t_layers != "none":
            layers = []
            last = self.p_dim
            for n in map(lambda x: int(x), proj_head_t_layers.split('-')[:-1]):
                layers.append(torch.nn.Linear(last, n, bias=False))
                layers.append(torch.nn.BatchNorm1d(n))
                layers.append(torch.nn.ReLU())
                last = n

            layers.append(torch.nn.Linear(last, int(proj_head_t_layers.split('-')[-1]), bias=False))
            layers.append(torch.nn.BatchNorm1d(int(proj_head_t_layers.split('-')[-1])))

            self.proj_head_t = torch.nn.Sequential(*layers)
            self.pp_dim = int(proj_head_t_layers.split('-')[-1])
        else:
            self.pp_dim = self.p_dim

        self.predictor_eq = None
        if self.predictor_eq != "none":
            layers = []
            last = self.y_dim+self.pp_dim
            for n in map(lambda x: int(x), predictor_eq_layers.split('-')[:-1]):
                layers.append(torch.nn.Linear(last, n, bias=False))
                layers.append(torch.nn.BatchNorm1d(n))
                layers.append(torch.nn.ReLU())
                last = n

            layers.append(torch.nn.Linear(last, self.y_dim, bias=False))
            layers.append(torch.nn.BatchNorm1d(self.y_dim, affine=False))

            self.predictor_eq = torch.nn.Sequential(*layers)
    
    def forward(self, images, params):
        h = self.resnet(images)

        z = self.proj_head_inv(h[h.shape[0]//3:])
        
        y = h if self.proj_head_eq is None else self.proj_head_eq(h)
        y0 = torch.cat([y[:y.shape[0]//3], y[:y.shape[0]//3]], dim=0)
        yt = y[y.shape[0]//3:]

        p = params if self.proj_head_t is None else self.proj_head_t(params)

        yt_hat = self.predictor_eq(torch.cat([y0, p], dim=1))

        return h, z, y0, yt, yt_hat
