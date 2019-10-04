import ml_metrics as metrics
import numpy as np
# numpy.matrix.flatten ??


def mapk(k, image):
    """
    Computes the mean average precision of an image for a k set of predictions
    Args:
        - k: number of predictions
        - image: image used to compute the k most similar images
    Returns:
        - mapk: mean average precision of the image for a k set of predictions
    """
    actual = [image]
    predicted = [k_retrieval(image)] # llista de llistes dels k predits
    return mmetrics.mapk(k, actual, predicted)


def compute_image_comparision(image_ref, image_pred):
    """
    Computes the precision, recall and f1 score between a binarized image and its prediciton
    Args:
        - image_ref: gound truth image as numpy.array
        - image_pred: predicted image as numpy.array
    Returns:
        - precision: precision of the predicted image in comparision to the original image
        - recall: recall of the predicted image in comparision to the original image
        - f1: f1 score of the predicted image in comparision to the original image
    """
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for i in range(image_ref.shape[0]):
        for j in range(image_ref.shape[1]):
            if(image_ref[i][j] and image_pred[i][j]):
                tp+=1.0
            elif(image_ref[i][j] and not(image_pred[i][j])):
                fn+=1.0
            elif(not(image_ref[i][j]) and image_pred[i][j]):
                fp+=1.0
            else:
                tn+=1.0
    precision = tp / (tp + fp)
    recall = tp /(tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def test_compute_image_comparision():
    a = np.array([[1,0,1],[1,0,1]])
    b = np.array([[1,0,1],[1,0,0]])
    return(compute_image_comparision(a,b))
