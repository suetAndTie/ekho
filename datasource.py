from sourcedataset import FileDataSource


class TextDataSource(FileDataSource):
    def __init__(self, data_root, speaker_id=None):
        self.data_root = data_root
        self.speaker_ids = None
        self.multi_speaker = False
        # If not None, filter by speaker_id
        self.speaker_id = speaker_id

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.multi_speaker = len(l) == 5
        texts = list(map(lambda l: l.decode("utf-8").split("|")[3], lines))
        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            # Filter by speaker_id
            # using multi-speaker dataset as a single speaker dataset
            if self.speaker_id is not None:
                indices = np.array(speaker_ids) == self.speaker_id
                texts = list(np.array(texts)[indices])
                self.multi_speaker = False
                return texts

            return texts, speaker_ids
        else:
            return texts

    def collect_features(self, *args):
        if self.multi_speaker:
            text, speaker_id = args
        else:
            text = args[0]
        global _frontend
        if _frontend is None:
            _frontend = getattr(frontend, hparams.frontend)
        seq = _frontend.text_to_sequence(text, p=hparams.replace_pronunciation_prob)

        if platform.system() == "Windows":
            if hasattr(hparams, 'gc_probability'):
                _frontend = None  # memory leaking prevention in Windows
                if np.random.rand() < hparams.gc_probability:
                    gc.collect()  # garbage collection enforced
                    print("GC done")

        if self.multi_speaker:
            return np.asarray(seq, dtype=np.int32), int(speaker_id)
        else:
            return np.asarray(seq, dtype=np.int32)
