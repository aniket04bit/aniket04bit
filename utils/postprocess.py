import torch
import numpy as np


class CTCLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            # Default English character set (alphanumeric)
            self.character_str = []
            # Add digits
            for i in range(10):
                self.character_str.append(str(i))
            # Add uppercase letters
            for i in range(26):
                self.character_str.append(chr(ord('A') + i))
            # Add lowercase letters  
            for i in range(26):
                self.character_str.append(chr(ord('a') + i))
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)

        if use_space_char:
            self.character_str.append(" ")

        dict_character = list(self.character_str)

        # Add blank token for CTC
        dict_character = ['<blank>'] + dict_character
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for CTC blank token


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            dict_character = list(self.character_str)

        if use_space_char:
            dict_character.append(" ")

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def encode(self, text):
        """Convert text to label indices."""
        if len(text) == 0 or len(text) > 25:
            return None
        if self.reverse:
            text = text[::-1]
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


def greedy_decode(predictions, blank_idx=0):
    """
    Greedy CTC decoding.
    
    Args:
        predictions: [B, T, C] tensor of log probabilities
        blank_idx: Index of blank token
    
    Returns:
        decoded: List of decoded sequences
    """
    B, T, C = predictions.shape
    
    # Get most likely characters at each timestep
    pred_indices = torch.argmax(predictions, dim=2)  # [B, T]
    
    decoded = []
    for b in range(B):
        # Remove consecutive duplicates and blanks
        sequence = []
        prev_idx = None
        
        for t in range(T):
            curr_idx = pred_indices[b, t].item()
            
            # Skip blanks
            if curr_idx == blank_idx:
                prev_idx = curr_idx
                continue
                
            # Skip consecutive duplicates
            if curr_idx != prev_idx:
                sequence.append(curr_idx)
            
            prev_idx = curr_idx
            
        decoded.append(sequence)
    
    return decoded


def beam_search_decode(predictions, beam_width=10, blank_idx=0):
    """
    Beam search CTC decoding.
    
    Args:
        predictions: [B, T, C] tensor of log probabilities
        beam_width: Beam width for search
        blank_idx: Index of blank token
    
    Returns:
        decoded: List of best decoded sequences
    """
    B, T, C = predictions.shape
    
    # Convert to log probabilities
    log_probs = torch.log_softmax(predictions, dim=2)
    
    decoded = []
    
    for b in range(B):
        # Initialize beam with empty sequence
        beam = [{'sequence': [], 'score': 0.0, 'last_char': None}]
        
        for t in range(T):
            candidates = []
            
            for beam_item in beam:
                for c in range(C):
                    score = beam_item['score'] + log_probs[b, t, c].item()
                    
                    if c == blank_idx:
                        # Blank - add to existing sequence
                        candidates.append({
                            'sequence': beam_item['sequence'][:],
                            'score': score,
                            'last_char': None
                        })
                    else:
                        # Character
                        new_sequence = beam_item['sequence'][:]
                        
                        # Only add if different from last character
                        if beam_item['last_char'] != c:
                            new_sequence.append(c)
                        
                        candidates.append({
                            'sequence': new_sequence,
                            'score': score,
                            'last_char': c
                        })
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beam = candidates[:beam_width]
        
        # Return best sequence
        best = max(beam, key=lambda x: x['score'])
        decoded.append(best['sequence'])
    
    return decoded