# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pascal_voc import pascal_voc

__sets = {}

for split in ['train', 'val', 'trainval', 'test']:
  __sets[split] = (lambda split=split, : pascal_voc(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
      raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
