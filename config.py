class BERTCONFIG:
    def __init__(self) -> None:
        self.SEED = 42

        self.EPOCHS = 2000
        self.BATCH_SIZE = 6

        self.EMBEDDING_DIM = 128
        self.SEQ_LENGTH = 512
        self.NUM_HEADS = 8
        self.NUM_LAYERS = 12

        self.EMBEDDING_DIM_LEN = [10, 14, 12, 12, 12, 18, 22, 12]
        self.NOTE_PROPERTIES = [
            "Octave",
            "Pitch",
            "Short Duration",
            "Medium Duration",
            "Long Duration",
            "Velocity",
            "Short Shift",
            "Long Shift",
        ]