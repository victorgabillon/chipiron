# import yaml
# import pickle
#
#
# class NN1:
#
#     def __init__(self, folder):
#         #   self.batch_forward = vmap(self.forward_pass, in_axes=(None, 0), out_axes=0)
#         self.num_classes = 2
#         self.key = random.PRNGKey(1)
#         self.params = self.initWeights(folder)
#
#        # self.jit_forward_pass = jit(forward_pass)
#
#     def initWeights(self, folder):
#
#         with open(r'runs/players/boardevaluators/NN1/' + folder + '/hyper_param.yaml')  as fileParam:
#             # print(fileParam.read())
#             hyper_param = yaml.load(fileParam, Loader=yaml.FullLoader)
#         layer_sizes = hyper_param['layer_sizes']
#         print(layer_sizes)
#
#         try:
#             with open('runs/players/boardevaluators/NN1/' + folder + '/param', 'rb')  as fileNNR:
#                 params = pickle.load(fileNNR)
#
#         except EnvironmentError:
#             params = initialize_mlp(layer_sizes, self.key)
#             with open('runs/players/boardevaluators/NN1/' + folder + '/param', 'wb')  as fileNNW:
#                 pickle.dump(params, fileNNW)
#
#         return params
#
#     def value_white(self, board):
#         input = board_to_nn_input(board)
#         output = forward_pass(self.params, input)
#         return output[0]
#
#
# def transform_board(board):
#     transform = [0] * 768
#     # print('ol', board.chessBoard)
#     for square in range(64):
#         piece = board.chess_board.piece_at(square)
#         if piece:
#             # print('p', square, piece.color, piece.piece_type)
#             piece_code = 6 * piece.color + (piece.piece_type - 1)
#             # print('dp', 64 * piece_code + square, 2 * piece.color - 1)
#             transform[64 * piece_code + square] = 2 * piece.color - 1
#         # transform[64 * piece_code + square] = 2 * piece.color - 1
#     return transform
#
#
# def board_to_nn_input(board):
#     transform = transform_board(board)
#     return board_to_nn_input_2(transform)
#
#
# def board_to_nn_input_2(transform):
#     transform_2 = jnp.zeros(768)
#     new_transform_2 = index_update(transform_2, index[:], transform)
#     return new_transform_2
#
#
# @jit
# def ReLU(x):
#     """ Rectified Linear Unit (ReLU) activation function """
#     return jnp.maximum(0, x)
#
#
# def initialize_mlp(sizes, key):
#     """ Initialize the weights of all layers of a linear layer network """
#     keys = random.split(key, len(sizes))
#
#     # Initialize a single layer with Gaussian weights -  helper function
#     def initialize_layer(m, n, key, scale=1e-2):
#         w_key, b_key = random.split(key)
#         return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
#
#     return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
#
# @jit
# def relu_layer(params, x):
#     """ Simple ReLu layer for single sample """
#     return ReLU(jnp.dot(params[0], x) + params[1])
#
# @jit
# def forward_pass(params, in_array):
#     """ Compute the forward pass for each example individually """
#     activations = in_array
#
#     # Loop over the ReLU hidden layers
#     for w, b in params[:-1]:
#         activations = relu_layer([w, b], activations)
#
#     # Perform final trafo to logits
#     final_w, final_b = params[-1]
#     logits = jnp.dot(final_w, activations) + final_b
#     return logits - logsumexp(logits)
#
# @jit
# def one_hot(x, k, dtype=jnp.float32):
#     """Create a one-hot encoding of x of size k """
#     return jnp.array(x[:, None] == jnp.arange(k), dtype)
#
# # def loss( params, in_arrays, targets):
# #     """ Compute the multi-class cross-entropy loss """
# #     preds = batch_forward(params, in_arrays)
# #     return -np.sum(preds * targets)
# #
# # def accuracy( params, data_loader):
# #     """ Compute the accuracy for a provided dataloader """
# #     acc_total = 0
# #     for batch_idx, (data, target) in enumerate(data_loader):
# #         images = np.array(data).reshape(data.size(0), 28 * 28)
# #         targets = one_hot(np.array(target), num_classes)
# #
# #         target_class = np.argmax(targets, axis=1)
# #         predicted_class = np.argmax(batch_forward(params, images), axis=1)
# #         acc_total += np.sum(predicted_class == target_class)
# #     return acc_total / len(data_loader.dataset)
