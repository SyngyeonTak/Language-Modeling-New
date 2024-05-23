# import some packages you need here
import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        # write your codes here
        with open(input_file, 'r') as f:
            text = f.read()

        # Character creation and putting characters in dictionaries of char2idx and idx2char
        self.chars = sorted(set(text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}

        # convert all characters in the text to indices
        self.data = [self.char_to_idx[ch] for ch in text]

        # Split the data into chunks of sequence length 30. 
        self.seq_length = 30 
        self.num_chunks = len(self.data) // self.seq_length


    def __len__(self):
        # write your codes here
        return self.num_chunks - 1

    def __getitem__(self, idx):

        # write your codes here
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        # Input sequence
        # elements will be indices in each input_seq_list
        input_seq = self.data[start_idx:end_idx]
        # Target sequence (shifted by one character)
        target_seq = self.data[start_idx + 1:end_idx + 1]

        input = torch.tensor(input_seq, dtype=torch.long)
        target = torch.tensor(target_seq, dtype=torch.long)

        return input, target

if __name__ == '__main__':
    # write test codes to verify your implementations
    dataset = Shakespeare('./datasets/shakespeare_train.txt')
    print("Dataset length:", len(dataset))
    print("Sample input and target:", dataset[0])