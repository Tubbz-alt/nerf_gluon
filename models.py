from mxnet import nd
from mxnet.gluon import nn


class PaperNeRFModel(nn.HybridBlock):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """
    def __init__(self, num_layers=8, hidden_size=256, skip_connect_every=4, num_encoding_fn_xyz=6,
                 num_encoding_fn_dir=4, include_input_xyz=True, include_input_dir=True, use_viewdirs=True):
        super(PaperNeRFModel, self).__init__()
        self.num_layers = num_layers
        self.num_dir_layers = 1
        self.hidden_size = hidden_size
        self.skip_connect_every = skip_connect_every
        self.use_viewdirs = use_viewdirs

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layers_xyz = []
        for i in range(num_layers):
            if i % self.skip_connect_every == 0 and i > 0 and i < num_layers - 1:
                self.layers_xyz.append(nn.Dense(hidden_size)) # , in_units=self.dim_xyz + hidden_size
            else:
                self.layers_xyz.append(nn.Dense(hidden_size)) # , in_units=hidden_size
            self.register_child(self.layers_xyz[-1])
        self.fc_feat = nn.Dense(hidden_size) # , in_units=hidden_size
        self.fc_alpha = nn.Dense(1) # , in_units=hidden_size

        self.layers_dir = []
        if self.use_viewdirs:
            self.layers_dir.append(nn.Dense(hidden_size)) # , in_units=hidden_size + self.dim_dir
        else:
            self.layers_dir.append(nn.Dense(hidden_size)) # , in_units=hidden_size
        self.register_child(self.layers_dir[-1])
        for i in range(self.num_dir_layers - 1):
            self.layers_dir.append(nn.Dense(hidden_size // 2)) # , in_units=hidden_size // 2
            self.register_child(self.layers_dir[-1])
        self.fc_rgb = nn.Dense(3) # , in_units=hidden_size // 2
        self.act = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        for i in range(self.num_layers):
            if i % self.skip_connect_every == 0 and i > 0 and i < self.num_layers - 1:
                x = self.layers_xyz[i](F.concat(*[xyz, x], dim=-1))
            else:
                x = self.layers_xyz[i](x)
            x = self.act(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](F.concat(*[feat, dirs], dim=-1))
        else:
            x = self.layers_dir[0](feat)
        x = self.act(x)
        for i in range(1, self.num_dir_layers):
            x = self.layers_dir[i](x)
            x = self.act(x)
        rgb = self.fc_rgb(x)
        return F.concat(*[rgb, alpha], dim=-1)


class FlexibleNeRFModel(nn.HybridBlock):
    def __init__(self, num_layers=4, hidden_size=128, skip_connect_every=4, num_encoding_fn_xyz=6,
                 num_encoding_fn_dir=4, include_input_xyz=True, include_input_dir=True, use_viewdirs=True):
        super(FlexibleNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = nn.Dense(hidden_size, in_units=self.dim_xyz)
        self.layers_xyz = []
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(nn.Dense(hidden_size, in_units=self.dim_xyz + hidden_size,
                                                weight_initializer=wght_init))
            else:
                self.layers_xyz.append(nn.Dense(hidden_size, in_units=hidden_size))
            self.register_child(self.layers_xyz[-1])

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = []
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(nn.Dense(hidden_size // 2, in_units=self.dim_dir + hidden_size,
                                            weight_initializer=wght_init))
            self.register_child(self.layers_dir[-1])

            self.fc_alpha = nn.Dense(1, in_units=hidden_size)
            self.fc_rgb = nn.Dense(3, in_units=hidden_size // 2)
            self.fc_feat = nn.Dense(hidden_size, in_units=hidden_size)
        else:
            self.fc_out = nn.Dense(4, in_units=hidden_size)

        self.act = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
            view = None

        x = self.layer1(xyz)
        x = self.act(x)
        for i in range(len(self.layers_xyz)):
            if i % self.skip_connect_every == 0 and i > 0 and i != len(self.layers_xyz) - 1:
                x = F.concat(*[x, xyz], dim=-1)
            x = self.layers_xyz[i](x)
            x = self.act(x)

        if self.use_viewdirs:
            feat = self.fc_feat(x)
            feat = self.act(feat)
            alpha = self.fc_alpha(x)
            x = F.concat(*[feat, view], dim=-1)
            for layer in self.layers_dir:
                x = layer(x)
                x = self.act(x)
            rgb = self.fc_rgb(x)
            return F.concat(*[rgb, alpha], dim=-1)
        else:
            out = self.fc_out(x)
            return out