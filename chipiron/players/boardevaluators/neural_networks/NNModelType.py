from enum import Enum


class NNModelType(str, Enum):
    NetP1 = "p1"
    NetP2 = "p2"
    NetPP1 = "pp1"
    NetPP2 = "pp2"
    NetPP2D2 = "pp2d2"
    NetPP2D2_2 = "pp2d2_2"
    NetPP2D2_2_LEAKY = "pp2d2_2_leaky"
    NetPP2D2_2_RRELU = "pp2d2_2_rrelu"
    NetPP2D2_2_PRELU = "pp2d2_2_prelu"
    TransformerOne = "transformerone"
