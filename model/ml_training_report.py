from dataclasses import dataclass

@dataclass
class MLTrainingReport:
    """Class for keeping track of ml training"""
    loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    val_accuracy: float = 0.0
    precision: float = 0.0
    val_precision: float = 0.0
    recall: float = 0.0
    val_recall: float = 0.0
    true_positives: float = 0.0
    val_true_positives: float = 0.0
    true_negatives: float = 0.0
    val_true_negatives: float = 0.0
    false_positives: float = 0.0
    val_false_positives: float = 0.0
    false_negatives: float = 0.0
    val_false_negatives: float = 0.0
    number_of_records: int = 0
    training_time: int = 0 # in seconds
    timestamp: int = 0
    correct_predictions: int = 0
    incorrect_predictions: int = 0
