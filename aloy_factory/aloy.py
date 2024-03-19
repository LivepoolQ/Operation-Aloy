"""
@Author: Ziqian Zou
@Date: 2024-03-18 21:23:32
@LastEditors: Ziqian Zou
@LastEditTime: 2024-03-18 21:53:35
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2024 Ziqian Zou, All Rights Reserved.
"""
from qpid.args import DYNAMIC, STATIC, TEMPORARY, Args
from qpid.model import Model
from qpid.training import Structure
from qpid.model.layers import LinearLayerND

class AloyArgs(Args):
    """
    First build args of operation-aloy
    """
    @property
    def number_fire_arrow(self) -> int:
        """
        The number of fire arrows used in the hunter bow.
        """
        return self._arg('number_fire_arrow', 20, DYNAMIC)
    
class AloyModel(Model):
    def __init__(self, Args: Args, structure=None, *args, **kwargs):
        super().__init__(Args, structure, *args, **kwargs)
        self.aloy_args = self.args.register_subargs(AloyArgs, 'aloyargs')
        self.linear = LinearLayerND(
            obs_frames=self.args.obs_frames, 
            pred_frames=self.args.pred_frames,
            diff=self.aloy_args.number_fire_arrow/20
        )

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        return self.linear(inputs[0])
    
class Aloy(Structure):

    is_trainable = False

    def create_model(self):
        self.model = AloyModel(self.args, structure=self)