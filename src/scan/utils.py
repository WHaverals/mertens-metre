import re
import json
import numpy as np


def preprocess_line(text):
    """preprocess text for scansion model
    
    args:
        text: input text string
        
    returns:
        preprocessed text (lowercase, stripped non-alpha except spaces)
    """
    if not text or text == "None":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def pred_to_classes(predictions):
    """convert softmax probabilities to class indices
    
    args:
        predictions: numpy array of shape (n_samples, max_len, 4)
        
    returns:
        numpy array of shape (n_samples, max_len) with class indices
    """
    return np.argmax(predictions, axis=-1)


def jsonify(text, classes):
    """convert predictions to JSON format
    
    args:
        text: Original preprocessed text string
        classes: Array of class indices for each position 
        includes BOS/EOS at positions 0 and len(text)+1 (!!! important !!!)
        
    returns:
        JSON string with words, syllables, and stress labels
    """
    if not text:
        return json.dumps({"line": "", "scanned": []}, ensure_ascii=False)
    
    text_len = len(text)
    classes = classes[1:text_len+1]  # Skip BOS (pos 0), take up to text_len characters
    
    words = text.split()
    scanned = []
    text_pos = 0
    
    for word in words:
        syllables = []
        syllable = ""
        stress = 0
        
        for i, char in enumerate(word):
            pos = text_pos + i
            
            if pos >= len(classes) or pos >= text_len:
                break
            
            if classes[pos] == 0:
                break
            
            class_idx = classes[pos]
            
            if class_idx == 2:
                if syllable:
                    syllables.append({
                        "syllable": syllable,
                        "stress": stress
                    })
                syllable = char
                stress = 0
            elif class_idx == 3:
                if syllable:
                    syllables.append({
                        "syllable": syllable,
                        "stress": stress
                    })
                syllable = char
                stress = 1
            else:
                syllable += char
        
        if syllable:
            syllables.append({
                "syllable": syllable,
                "stress": stress
            })
        
        if not syllables:
            syllables.append({
                "syllable": word,
                "stress": 0
            })
        
        scanned.append({
            "token": word,
            "syllables": syllables
        })
        
        text_pos += len(word) + 1
    
    result = {
        "line": text,
        "scanned": scanned
    }
    
    return json.dumps(result, ensure_ascii=False)