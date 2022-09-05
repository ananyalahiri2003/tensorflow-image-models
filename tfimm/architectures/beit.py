"""
We provide an implementation and pretrained weights for BERT Pre-Training of Image 
Transformers (BEIT).
Paper: Rethinking Spatial Dimensions of Vision Transformers.
`[arXiv:2106.08254] <https://arxiv.org/abs/2106.08254>`_.
Original pytorch code and weights from
`<https://github.com/microsoft/unilm/tree/master/beit>`_.
This code has been ported from the
`timm <https://github.com/rwightman/pytorch-image-models>`_ implementation.
"""

# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Modifications for timm by / Copyright 2020 Ross Wightman
# Copyright 2022 Martins Bruveris

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf

from tfimm.architectures.vit import ViTBlock
from tfimm.layers import interpolate_pos_embeddings_grid, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Model registry will add each entrypoint fn to this
__all__ = ["BERTPretrainImageTransformer", "BERTPretrainImageTransformerConfig"]

