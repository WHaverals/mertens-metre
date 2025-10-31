import json
import numpy as np

class SequenceVectorizer:
    """char-level sequence vectorizer for Middle Dutch verse scansion"""
    
    def __init__(self, vocab, max_len):
        self.vocab = vocab
        self.max_len = max_len
        self.idx2syll = {int(k): v for k, v in vocab['idx2syll'].items()}
        self.syll2idx = vocab['syll2idx']
    
    @classmethod
    def load(cls, path):
        """load vectorizer from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab, vocab['max_len'])
    
    def transform(self, texts):
        """transform text sequences to model input format
        
        args:
            texts: list of strings or single string
            
        returns:
            numpy array of shape (n_samples, max_len)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        sequences = []
        for text in texts:
            seq = [self.syll2idx.get('<BOS>', 1)]
            for char in text:
                idx = self.syll2idx.get(char, self.syll2idx.get('<UNK>', 3))
                seq.append(idx)
                if len(seq) >= (self.max_len - 1):
                    break
            seq.append(self.syll2idx.get('<EOS>', 2))
            
            pad_len = self.max_len - len(seq)
            seq.extend([self.syll2idx.get('<PAD>', 0)] * pad_len)
            
            sequences.append(seq)
        
        return np.array(sequences, dtype=np.int32)
        