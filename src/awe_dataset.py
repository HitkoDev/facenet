import os
import glob


class AWEDataset(object):

    def __init__(self, path):
        super().__init__()
        images = {}
        classes = []
        for d in os.listdir(path):
            c = d
            classes.append(c)
            dir = os.path.join(path, d)
            for f in glob.glob(dir + '/*.png'):
                if c not in images:
                    images[c] = []
                images[c].append({
                    "subject": d,
                    "src": f,
                    "mask": f[:-4] + '.npy',
                    "class": c
                })
        self.images = [images[k] for k in images]
        self.classes = classes
