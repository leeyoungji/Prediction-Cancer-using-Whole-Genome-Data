from torch.utils.data import DataLoader, Dataset
import torch
import json
import csv

class WGDs(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0 = 44900):
        """Create AG's News dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        self.l0 = l0
        self.loadAlphabet(alphabet_path)
        self.load(label_data_path)

    def __len__(self):
        print(len(self.label))
        return len(self.label)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y

    def loadAlphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase = True):
        self.label = []
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for index, row in enumerate(rdr):
                self.label.append(int(row[0]))
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()                
                self.data.append(txt)

        self.y = torch.LongTensor(self.label)
        print('max : ', max(len(w) for w in self.data))

    def oneHotEncode(self, idx):
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char)!=-1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def getClassWeight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples/float(self.label.count(c)) for c in label_set]    
        return class_weight, num_class

if __name__ == '__main__':
    
    label_data_path = 'data_/csv/val_set12.csv'
    alphabet_path = 'alphabet.json'

    train_dataset = WGDs(label_data_path, alphabet_path)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, drop_last=False)

    # size = 0
    for i_batch, sample_batched in enumerate(train_loader):
        if i_batch == 0:
            print(sample_batched[0][0][0].shape)
