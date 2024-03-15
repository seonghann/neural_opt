from .graph_encoder.schnet import SchNetEncoder as SchNet
# from .graph_encoder.leftnet import LEFTNet as LeftNet

EncoderDict = {
    "schnet": SchNet,
    # "leftnet": LeftNet,
}
