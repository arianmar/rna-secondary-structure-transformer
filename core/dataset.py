import torch
from torch.utils.data import Dataset


class RNADataset(Dataset):
    def __init__(self, items, base2id, struct2id):
        self.items = items
        self.base2id = base2id
        self.struct2id = struct2id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, struct = self.items[idx]
        x = torch.tensor([self.base2id[ch] for ch in seq], dtype=torch.long)
        y = torch.tensor([self.struct2id[ch] for ch in struct], dtype=torch.long)
        return x, y


class PermutedRNADataset(Dataset):
    def __init__(self, items, perm, base2id, struct2id):
        self.items = items
        self.perm = perm
        self.base2id = base2id
        self.struct2id = struct2id

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, idx):
        seq, struct = self.items[self.perm[idx]]
        x = torch.tensor([self.base2id[ch] for ch in seq], dtype=torch.long)
        y = torch.tensor([self.struct2id[ch] for ch in struct], dtype=torch.long)
        return x, y


def make_collate(pad_x: int, pad_y: int):
    def collate(batch):
        xs, ys = zip(*batch)
        T = max(x.size(0) for x in xs)
        B = len(xs)

        x_pad = torch.full((B, T), pad_x, dtype=torch.long)
        y_pad = torch.full((B, T), pad_y, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.bool)

        for i, (x, y) in enumerate(zip(xs, ys)):
            L = x.size(0)
            x_pad[i, :L] = x
            y_pad[i, :L] = y
            mask[i, :L] = True

        return x_pad, y_pad, mask

    return collate