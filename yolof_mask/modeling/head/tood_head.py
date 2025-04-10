EPS = 1e-12

class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8, conv_cfg=None, norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.la_conv1 = nn.Conv2d( self.in_channels,  self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d( self.in_channels // la_down_rate,  self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.norm_cfg is None)

    def init_weights(self):
        normal_init(self.la_conv1, std=0.001)
        normal_init(self.la_conv2, std=0.001)
        self.la_conv2.bias.data.zero_()
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs.
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
                          self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat

class TOODHead(AnchorHead):
    """TOOD: Task-aligned One-stage Object Detection.

    TOOD uses Task-aligned head (T-head) and is optimized by Task Alignment
    Learning (TAL).

    todo: list link of the paper.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        stacked_convs=4,
        conv_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        num_dcn_on_head=0,
        anchor_type='anchor_free',
        initial_loss_cls=dict(
            type='TaskAlignedLoess',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        **kwargs
    ):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_dcn_on_head = num_dcn_on_head
        self.anchor_type = anchor_type
        self.epoch = 0 # which would be update in head hook!
        super(TOODHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.initial_epoch = self.train_cfg.initial_epoch
            self.initial_assigner = build_assigner(self.train_cfg.initial_assigner)
            self.initial_loss_cls = build_loss(initial_loss_cls)
            self.alingment_assigner = build_assigner(self.train_cfg.assigner)
            self.alpha = self.train_cfg.alpha
            self.beta = self.train_cfg.beta
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i < self.num_dcn_on_head:
                conv_cfg = dict(type='DCNv2', deform_groups=4)
            else:
                conv_cfg = self.conv_cfg
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8, self.conv_cfg, self.norm_cfg)
        self.reg_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8, self.conv_cfg, self.norm_cfg)

        self.tood_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.tood_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)

        self.cls_prob_conv1 = nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1)
        self.cls_prob_conv2 = nn.Conv2d(self.feat_channels // 4, 1, 3, padding=1)
        self.reg_offset_conv1 = nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1)
        self.reg_offset_conv2 = nn.Conv2d(self.feat_channels // 4, 4 * 2, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.anchor_generator.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.inter_convs:
            normal_init(m.conv, std=0.01)

        self.cls_decomp.init_weights()
        self.reg_decomp.init_weights()

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.tood_cls, std=0.01, bias=bias_cls)
        normal_init(self.tood_reg, std=0.01)

        normal_init(self.cls_prob_conv1, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_prob_conv2, std=0.01, bias=bias_cls)
        normal_init(self.reg_offset_conv1, std=0.001)
        normal_init(self.reg_offset_conv2, std=0.001)
        self.reg_offset_conv2.bias.data.zero_()