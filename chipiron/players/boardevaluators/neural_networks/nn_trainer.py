
class NNPytorchTrainer:

    def __init__(self, path_to_origin_folder, file_path, net, loss_criterion, optimizer):
        self.param_file = file_path
        self.path_to_origin_folder = path_to_origin_folder
        self.net = net
        self.criterion = loss_criterion,
        self.optimizer = optimizer

    def train_one_example(self, input_layer, target_value, target_input_layer):

        if target_value is None:
            assert (target_input_layer is not None)
            self.net.eval()
            # print('**', target_input_layer)
            real_target_value = 1 - self.net(target_input_layer)
            self.net.train()
        else:
            assert (target_input_layer is None)
            real_target_value = target_value

        self.net.train()
        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)
        target_min_max_value_player_to_move = torch.tensor([real_target_value])
        loss = self.criterion(prediction_with_player_to_move_as_white, target_min_max_value_player_to_move)
        loss.backward()
        self.optimizer.step()

        # Save new params
        new_state_dict = {}
        for key in self.net.state_dict():
            new_state_dict[key] = self.net.state_dict()[key].clone()

        # print('after')
        if random.random() < 0.01:
            self.print_param()
        # print('after')

        try:
            with open('chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.param_file, 'wb') as fileNNW:
                # print('ddfff', fileNNW)
                torch.save(self.net.state_dict(), fileNNW)
        except KeyboardInterrupt:
            with open('chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.param_file, 'wb') as fileNNW:
                # print('ddfff', fileNNW)
                torch.save(self.net.state_dict(), fileNNW)
            exit(-1)


