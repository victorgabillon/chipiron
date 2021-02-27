from players.boardevaluators.NN4_pytorch import NN4Pytorch
import pandas as pd
from players.

folder = 'NN104'

nn= NN4Pytorch(folder)


df= pd.read_pickle('chipiron/data/test50')

for index, row in df.iterrows():
    print(row)
    fen = row['fen']
    board= BOard
    input_layer=
    target_value_0_1=d
    target_input_layer
    nn.train_one_example( input_layer, target_value_0_1, target_input_layer):