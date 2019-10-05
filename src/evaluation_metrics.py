import ml_metrics as metrics
import numpy as np


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
    predicted = [k_retrieval(image)]  # llista de llistes dels k predits
    return mmetrics.mapk(k, actual, predicted)


def global_mapk(k, images):
    """
    Computes the mean of the mean average precision of a list of images for a k set of predictions of each image
    Args:
        - k: number of predictions
        - images: list of images used to compute the k most similar images
    Returns:
        - mean_mapk: mean of the mean average precision of the images for a k predictions
    """
    mean_mapk = 0.0
    for i in images:
        mean_mapk += mapk(k, i)
    return mean_mapk / float(len(images))


def binarized_image_comparision(image_ref, image_pred):
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
            if (image_ref[i][j] and image_pred[i][j]):
                tp += 1.0
            elif (image_ref[i][j] and not (image_pred[i][j])):
                fn += 1.0
            elif (not (image_ref[i][j]) and image_pred[i][j]):
                fp += 1.0
            else:
                tn += 1.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def global_binarized_image_comparisions(refs_images, preds_images):
    """
    Computes the precision, recall and f1 score between a list of binarized image and its predicitons
    Args:
        - refs_images: list of gound truth images
        - preds_images: list of predicted images
    Returns:
        - precision: mean of the precision of the predicted images in comparision to the original images
        - recall: mean of the recall of the predicted images in comparision to the original images
        - f1: mean of the f1 score of the predicted images in comparision to the original images
    """
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for i in range(len(refs_images)):
        precision_aux, recall_aux, f1_aux = compute_image_comparision(
            refs_images[i], preds_images[i])
        precision += precision_aux
        recall += recall_aux
        f1 += f1_aux
    return precision / float(len(refs_images)), recall / float(
        len(refs_images)), f1 / float(len(refs_images))


#TODO: erease
def test_compute_image_comparision():
    a = np.array([[1, 0, 1], [1, 0, 1]])
    aa = np.array([[1, 0, 1], [1, 0, 0]])
    b = np.array([[1, 0, 1], [1, 0, 0]])
    return (global_binarized_image_comparisions([a, aa], [b, b]))
