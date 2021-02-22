from games.play_one_game import PlayOneGame

class playGames():

    def __init__(self,games):

        self.games = games

    def play(self):

        for i in range(self.games):
            oneGame =  PlayOneGame(True)
            oneGame.play()
