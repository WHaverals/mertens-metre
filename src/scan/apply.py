from keras.models import load_model
from .vectorization import SequenceVectorizer


def load_scanner(model_path, vectorizer_path):
    """load scansion model and vectorizer
    
    args:
        model_path: Path to scanner.keras file
        vectorizer_path: Path to vectorizer.json file
        
    returns:
        tuple of (model, vectorizer)
    """
    vectorizer = SequenceVectorizer.load(vectorizer_path)
    model = load_model(model_path)
    return model, vectorizer


def predict_batch(model, vectorizer, texts, batch_size=32):
    """predict scansion for a batch of texts
    
    args:
        model: Loaded Keras model
        vectorizer: SequenceVectorizer instance
        texts: List of preprocessed text strings
        batch_size: Batch size for prediction
        
    returns:
        List of class index arrays
    """
    if not texts:
        return []
    
    X = vectorizer.transform(texts)
    predictions = model.predict(X, verbose=0, batch_size=batch_size)
    
    from .utils import pred_to_classes
    classes = pred_to_classes(predictions)
    
    return [classes[i] for i in range(len(texts))]
