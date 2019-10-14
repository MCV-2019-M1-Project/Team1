import ml_metrics as metrics
from k_retrieval import k_retrieval
import numpy as np


def mapk(gt_corresps_list, predicted_list, k):
    """
    Computes the mean average precision of an image for a k set of predictions
    Args:
        - gt_corresps: list of ground truth
        - predicted_list: list of predicted
        - k: maximum number of predicted
    Returns:
        - mapk: mean average precision of the image for a k set of predictions
    """

    k_retrieved_list = k_retrieval(predicted_list, k)
    k_retrieved_list = [k.db_im.image.image_name() for k in k_retrieved_list]
    return metrics.mapk([gt_corresps_list], [k_retrieved_list], k)


def global_mapk(gt_corresps_lists, predicted_lists, k):
    """
    Computes the mean of the mean average precision of a list of images for a k set of predictions of each image
    Args:
        - gt_corresps_lists: list of lists of ground truth
        - predicted_lists: list of lists of predicted
        - k: maximum number of predicted
    Returns:
        - mean_mapk: mean of the mean average precision of the images for a k predictions
    """

    mean_mapk = 0.0
    for gt_corresps_list, predicted_list in zip(gt_corresps_lists, predicted_lists):
        mean_mapk += mapk(gt_corresps_list, predicted_list, k)
    return mean_mapk / float(len(gt_corresps_lists))


def compute_image_comparision(image_ref, image_pred):
    """
    Computes the precision, recall and f1 score between a binarized image and its prediciton
    Args:
        - image_ref: gound truth image as an Image
        - image_pred: predicted image as an Image
    Returns:
        - precision: precision of the predicted image in comparision to the original image
        - recall: recall of the predicted image in comparision to the original image
        - f1: f1 score of the predicted image in comparision to the original image
    """
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for i in range(image_ref.img.shape[0]):
        for j in range(image_ref.img.shape[1]):
            if (image_ref.img[i][j][0] and image_pred[i][j]):
                tp += 1.0
            elif (image_ref.img[i][j][0] and not (image_pred[i][j])):
                fn += 1.0
            elif (not (image_ref.img[i][j][0]) and image_pred[i][j]):
                fp += 1.0
            else:
                tn += 1.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def global_compute_image_comparision(refs_images, preds_images):
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
    
def bb_intersection_over_union(predict_box, ground_truth_box):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(predict_box[0], ground_truth_box[0])
    y1 = max(predict_box[1], ground_truth_box[1])
    x2 = min(predict_box[2], ground_truth_box[2])
    y2 = min(predict_box[3], ground_truth_box[3])

    # compute the area of intersection
    intersect_area = (x2 - x1) * (y2 - y1)

    # compute the area of bounding boxes
    predict_area = (predict_box[2] - predict_box[0]) * (predict_box[3] - predict_box[1])
    ground_truth_area = (ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1])


    #intersection area divided by prediction and ground-truth area
    iou = intersect_area / float(predict_area  + ground_truth_area - intersect_area)


    return iou