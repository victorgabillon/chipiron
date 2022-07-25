class BoardModification:
    """
    object that describes the modification to a board from a move
    """

    def __init__(self):
        self.removals = set()
        self.appearances = set()

    def add_appearance(self, appearance):
        self.appearances.add(appearance)

    def add_removal(self, removal):
        self.removals.add(removal)
