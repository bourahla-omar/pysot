# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.sa.sa import sablock

SABLOCKS = {
              'sablock': sablock,
            }


def get_sa(name, **kwargs):
    return SABLOCKS[name](**kwargs)
