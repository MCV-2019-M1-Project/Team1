import ml_metrics as metrics
from sklearn.metrics import precision_recall_curve
# numpy.matrix.flatten ??


#MAP@K: mean average precision at k for a set of predictions
def mapk(k):
    actual = [] #Llista de llistes amb el valor a cada un
    predicted = [] # llista de llistes dels k predits
    return mmetrics.mapk(k, actual, predicted)

#Returns the precision, recall and f1 score of two images
def compute_image_comparision(image_ref, image_pred):
    precision, recall, _ = precision_recall_curve(image_ref.flatten(), image_pred.flatten())
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
