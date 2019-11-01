import pickle
import os
try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2
import ml_metrics as metrics
import datetime
import random
import numpy as np
from text_recognition import text_recognition
from background_remover import remove_background, get_background_and_text_mask
import re
from denoising import remove_salt_and_pepper_noise
from texture_descriptors import LBP, HOG
from image_descriptors import similarity_for_descriptors, best_color_descriptor
from text_detector import get_text_mask_BGR
from sift_descriptors import SIFT_descriptors_matcher, SIFT_method

#TODO #1: Validar background_removal y text_bounding_box detection

#TODO #2: Adaptar la pipeline para que soporte comparacion de keypoints

#TODO #3: Validar deinoising

## DUMMYS #####

def how_similar_dummy(desc1, desc2):
    return random.random()

def dummy_descriptor(image, **kwargs):
    return random.random()

###############

## TEXT-DESCRIPTORS ###

def sanitize(s):
    return re.sub(r'[^a-zA-Z ]','',s)

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def text_similarity(desc1, desc2):
    # Quito lo que no sean letras en minuscula/mayuscula y espacios
    desc1 = re.sub(r'[^a-zA-Z ]','',desc1)
    desc2 = re.sub(r'[^a-zA-Z ]','',desc2)
    if len(desc1) == len(desc2) == 0:
        return 0
    elif desc1 == desc2:
        return 1
    else:
        return 1.0/levenshtein(desc1, desc2)

def text_bbdd(image, **kwargs):
    image_filepath = kwargs["image_filepath"]
    num_as_str = os.path.split(image_filepath)[-1][-9:-4]
    with open(os.path.join("..","bbdd_text",num_as_str+".txt")) as f:
        try:
            author_or_nada = eval(f.read())[0]
        except:
            return ""
        return author_or_nada

def text_descriptor(image, **kwargs):
    return sanitize(text_recognition(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)))

#######################

## TEXTURE DESCRIPTORS ##

def generic_histogram_similarity(desc1, desc2):
    # Correlation es una similitud
    sim = similarity_for_descriptors(desc1, desc2, distance_method="correlation")
    return sim

def lbp_similarity(desc1, desc2):
    # Correlation es una similitud
    sim = similarity_for_descriptors(desc1, desc2, distance_method="x2_dist")
    return sim

def lbp_descriptor(image, **kwargs):
    #mask = get_text_mask_BGR(image)
    #kwargs.update(mask=mask)
    image_in_specific_space = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return LBP(image_in_specific_space,
               **kwargs
    )

def hog_similarity(desc1, desc2):
    return similarity_for_descriptors(desc1, desc2, distance_method="euclidean")

def hog_descriptor(image, **kwargs):
    return HOG(image)


### COLOR DESCRIPTORS ####################


def color_descriptor(image, **kwargs):
    mask = get_text_mask_BGR(image)
    kwargs.update(mask=mask)
    image_in_specific_space = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return best_color_descriptor(image_in_specific_space, **kwargs)

## KEYPOINTS DESCRIPTORS #################

def keypoints_descriptor(image, **kwargs):
    return SIFT_method(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), **kwargs)

def keypoints_similarity(desc1, desc2):
    """
    Similarity is based in the number of matches
    """
    num_matches = len(SIFT_descriptors_matcher(desc1, desc2, thres_dist=MIN_DIST_TO_BE_MATCH))
    return num_matches

#########################

## PREPROCESSING ##########

def apply_denoising(im):
    return remove_salt_and_pepper_noise(im)

def apply_preprocessing(image_filepath, preprocessing, no_bg, single_sure):
    im = cv2.imread(image_filepath)
    #im = apply_denoising(im)
    return [im] if no_bg else remove_background(im, single_sure=single_sure)

def merge_similarities(sim_by_desc, similarity_threshold=None):
    """
    :param sim_by_desc: { desc_name: [a, b, c], ... } being a = similarity of bbdd_item 0, b =  = similarity of bbdd_item 1, ...
    :param similarity_threshold: for the KEYPOINTS_DESCRIPTOR, the number of required matches for the most matched image, in order to not declare the query as -1
    :return: the final merged order: [index_at_bbdd_top1_similar, index_at_bbdd_top2_similar, ...]
    """

    def normalize(v):
        "escala los valores de v para ponerlos entre 0 y 1"
        return list((v - np.min(v))/np.ptp(v))

    if similarity_threshold is not None and len(sim_by_desc) == 1 and 'key' in list(sim_by_desc.keys())[0]:
        print("\t\tApplying similarity threshold on "+str(similarity_threshold)+" matches.")
        #Assume one single descriptor and desc_name = keypoints
        matches_per_bbdd_item = list(sim_by_desc.values())[0]
        if max(matches_per_bbdd_item) < similarity_threshold:
            return [-1]


    scaled_sim_by_desc = {k: normalize(sim_by_desc[k]) for k in sim_by_desc.keys()}
    merged_similarity = [0.0] * len(list(sim_by_desc.values())[0])
    for sim_vec in scaled_sim_by_desc.values():
        for i, sim in enumerate(sim_vec):
            merged_similarity[i] += sim

    final_order = [t[0] for t in sorted(enumerate(merged_similarity), key=lambda x: x[1], reverse=True)]
    return final_order

def order_by_similarity(bbdd_descriptors, query_descriptor, descriptors_sim, similarity_threshold=None):
    """
    :param bbdd_descriptors: [dict(im_num -> dict(desc_name -> desc_value)), ... ]
    :param query_descriptor: dict(desc_name -> desc_value)
    :param descriptors_sim: { desc_name: fun(obj, obj) -> sim , ...}
    :return: indexes of the bbdd ordered by similarity
    """
    # Para cada descriptor un orden. Despues, merge.
    sim_by_desc = {}

    for desc_name, desc_obj in query_descriptor.items():
        print("\tComputing similarity for descriptor "+desc_name)
        # De la base de datos, tengo que coger la imagen "i", y dentro de esta SIEMPRE la 0, porque la bbdd solo tiene un cuadro
        sim_list = [descriptors_sim[desc_name](desc_obj, bbdd_descriptors[i][0][desc_name]) for i in sorted(bbdd_descriptors.keys())]
        sim_by_desc[desc_name] = sim_list[:]

    final_order = merge_similarities(sim_by_desc, similarity_threshold)
    print("\tMerged predictions:",final_order[:10])
    return final_order

def compute_similarity_all_queryset(bbdd_descriptors, query_descriptors, descriptors_sim, k_closest=20, similarity_threshold=None):
    """
    :param query_descriptors: [im_num: { desc_name: fun(I) -> obj , ... }, ...]
    :param bbdd_descriptors:  [im_num: { desc_name: fun(I) -> obj , ... }, ...]
    :param descriptors_sim: { desc_name: fun(obj, obj) -> sim , ...}

    """
    predictions = []
    for query_key in sorted(query_descriptors.keys()): # Itero sobre todas las queries
        print("Computing similarity for query #"+str(query_key))
        descriptors_list = query_descriptors[query_key] # Lista que contiene un diccionario de descriptores por cada cuadro encontrado
        query_image_result = []
        for query_descriptor in descriptors_list: # query_descriptor es un { desc_name: fun(I) -> obj , ... }
            query_image_result.append(
                    order_by_similarity(bbdd_descriptors,
                                        query_descriptor,
                                        descriptors_sim,
                                        similarity_threshold)[:k_closest]
            )
        predictions.append(query_image_result[:])
    return predictions

def get_descriptors_given_query_dir(descriptors_definition,
                                    folder,
                                    recompute=True,
                                    preprocessing=None,
                                    no_bg=False,
                                    single_sure=False,
                                    kwargs_for_descriptors=None,
                                    ):
    """
    Abstraction layer that either obtains a cached version of the bbdd descriptors or recomputes and stores them
    :param descriptors_definition: dict of functions that take an image and return an obj
    :param folder: folder containing the images that we want to describe
    :param recompute: tells if descriptors must be recomputed
    :param preprocessing: list of preprocessing functions that need to be applied
    :param single_sure: ONLY CONSIDERED IF preprocessing is not None
    :param no_bg: ONLY CONSIDERED IF preprocessing is not None
    :return: dict(im_num -> lista de diccionarios(descriptor_name -> descriptor_value), 1 para cada cuadro encontrado)
    """
    print("Getting descriptors for directory: ", folder)
    if not recompute:
        try:
            with open(os.path.join(folder,"descriptors.pkl"), 'rb') as handle:
                print("\tGetting cached at "+os.path.join(folder,"descriptors.pkl"))
                ds = pickle.load(handle)
                return ds
        except FileNotFoundError:
            pass

    print("\tNo cached version found. Recomputing...")

    folder_images_paths = [os.path.join(folder, image_filename)
                          for image_filename in
                          sorted(filter(
                                  lambda f: f.endswith('.jpg'), os.listdir(folder)))]

    # Para cada número de imagen, una lista que acabará conteniendo un diccionario por cada cuadro contenido en la imagen
    descriptors = {k: [{}, {}] for k in [int(os.path.split(x)[-1][-9:-4]) for x in folder_images_paths]}

    for image_filepath in folder_images_paths:
        print("Current image: "+image_filepath)
        im_num = int(os.path.split(image_filepath)[-1][-9:-4])
        if preprocessing is not None:
            list_containing_one_or_two_images = apply_preprocessing(image_filepath,
                                                                    preprocessing,
                                                                    no_bg,
                                                                    single_sure)
        else:
            list_containing_one_or_two_images = [cv2.imread(image_filepath)]

        print("\t"+str(len(list_containing_one_or_two_images))+ " paintings were found in the image")

        if kwargs_for_descriptors is None: kwargs_for_descriptors = {}
        for descriptor_name, descriptor_func in descriptors_definition.items():
            if descriptor_name in kwargs_for_descriptors:
                kwargs_for_this_descriptor = kwargs_for_descriptors[descriptor_name]
            else:
                kwargs_for_this_descriptor = {}
            if descriptor_func == text_bbdd:
                kwargs_for_this_descriptor.update(image_filepath=image_filepath)
            #try:
            for i, im in enumerate(list_containing_one_or_two_images):
                print("\tApplying descriptor " + descriptor_name + " to painting #"+str(i)+"... ",end="")
                descriptors[im_num][i][descriptor_name] = descriptor_func(im, **kwargs_for_this_descriptor)
                print("DONE")
            #except BaseException as e:
            #    print(e)

        # Cleaning empty dicts for that images that contained only one painting
    descriptors = {k: [x for x in descriptors[k] if len(x) > 0] for k in descriptors.keys()}
    return descriptors

def get_map_at_several_ks(query_dir, ks,
                          descriptors,
                          descriptors_sim,
                          preprocesses=None,
                          no_bg=False,
                          single_sure=False,
                          submission=False,
                          recompute_bbdd_descriptors=True,
                          recompute_query_descriptors=True,
                          kwargs_for_descriptors=None,
                          similarity_threshold=None
                          ):
    """
    :param query_dir: directory containing the query images

    :param ks: list containing all the ks for computing the map@k: Example: [1,5,10]

    :param descriptors: { desc_name: fun(I) -> obj , ... }

    :param descriptors_sim: { desc_name: fun(obj, obj) -> sim , ...}

    :param preprocesses: list of functions that take an image and return an image or list of images

    :param no_bg: boolean that says if images need de-backgrounding

    :param single_sure: boolean that says if the dataset contain only single images

    :param submission: boolean that tells if the execution is for submission (no GT)

    :param recompute_bbdd_descriptors: tells if descriptors of the bbdd must be recomputed

    :return: Diccionario con las K y map@k. Ejemplo: dict(1: 0.6, 3:0.8, 10:0.95)
    """

    try:
        bd_desc = None
        if "text" in descriptors:
            # Tengo que considerar que la manera de obtener el descriptor texto en las imagenes de la bbdd es diferente
            # ya que se obtiene de fichero
            bd_desc = descriptors.copy()
            bd_desc["text"] = text_bbdd

        bbdd_descriptors = get_descriptors_given_query_dir(bd_desc if "text" in descriptors else descriptors,
                                                           os.path.join("..","bbdd"),
                                                           recompute_bbdd_descriptors,
                                                           kwargs_for_descriptors=kwargs_for_descriptors)
    except Exception as e:
        print(" **************** Exception ocurred while getting the bbdd descriptors **************** ")
        print(e)
        raise e

    else:
        if recompute_bbdd_descriptors:
            with open(os.path.join("..","bbdd","descriptors.pkl"), "wb") as f:
                pickle.dump(bbdd_descriptors, f)
            print("Written bbdd descriptors at " + os.path.join("..","bbdd","descriptors.pkl"))

    try:
        query_descriptors = get_descriptors_given_query_dir(descriptors,
                                                            query_dir,
                                                            recompute_query_descriptors,
                                                            preprocessing=preprocesses,
                                                            no_bg=no_bg,
                                                            single_sure=single_sure,
                                                            kwargs_for_descriptors=kwargs_for_descriptors)
    except Exception as e:
        print("**************** Exception ocurred while getting the query descriptors **************** ")
        print(e)
        raise e
    else:
        if recompute_query_descriptors:
            with open(os.path.join(query_dir,"descriptors.pkl"), "wb") as f:
                pickle.dump(query_descriptors, f)
            print("Written query descriptors at "+os.path.join(query_dir,"descriptors.pkl"))

    predictions = compute_similarity_all_queryset(bbdd_descriptors,
                                                  query_descriptors,
                                                  descriptors_sim,
                                                  k_closest=max(ks),
                                                  similarity_threshold=similarity_threshold)

    if submission:
        with open(os.path.join(query_dir, "result.pkl"), "wb") as f:
            pickle.dump(predictions, f)
        print("Dumped submission results at ",os.path.join(query_dir, "result.pkl"))

    if not submission:
        with open(os.path.join(query_dir,"gt_corresps.pkl") ,'rb') as handle:
            actual = pickle.load(handle)

        map_ = {k: 0.0 for k in ks}

        number_of_paintings_queried = 0
        number_of_misidentified_number_of_paintings = 0
        for image_prediction_list, gt_list in zip(predictions, actual):
            if len(image_prediction_list) == len(gt_list): # El numero de cuadros detectados es el bueno
                for single_prediction, single_gt in zip(image_prediction_list, gt_list):
                    for k in ks:
                        if k > len(single_prediction) and single_prediction != [-1]: print("WARNING !!!!!!!!!!! - k is larger than the predictions")
                        map_[k] += metrics.apk([single_gt], single_prediction, k)
                    number_of_paintings_queried += 1

            elif len(image_prediction_list) < len(gt_list): # Habia 2 cuadros y solo hemos detectado uno
                number_of_paintings_queried += 2
                number_of_misidentified_number_of_paintings += 1

            elif len(image_prediction_list) > len(gt_list): # Habia 1 cuadro y hemos detectado 2
                number_of_paintings_queried += 1
                number_of_misidentified_number_of_paintings += 1

        map_ = {k : map_[k]/number_of_paintings_queried for k in map_.keys()}

        print()

        print("number_of_paintings_queried: ", number_of_paintings_queried)
        print("number_of_misidentified_number_of_paintings: ",number_of_misidentified_number_of_paintings)

        print()

        # with open(os.path.join(query_dir, "map_at_k"+datetime.datetime.now().isoformat()+".txt"), "w") as f:
        for k, m in sorted(map_.items()):
        #         f.write("map@"+str(k)+": "+"%.3f" % m+"\n")
            print("map@"+str(k)+": "+"%.3f" % m+"\n")
        return map_

def color_pipeline():
    # Solo color
    get_map_at_several_ks(
            query_dir=os.path.join("..", "queries", "qsd1_w4"),
            ks=[1, 5, 10],
            descriptors={"color": best_color_descriptor,
                         },
            descriptors_sim={"color": generic_histogram_similarity,
                             },
            preprocesses=True,
            no_bg=False,
            recompute_bbdd_descriptors=False,
            recompute_query_descriptors=True,
            kwargs_for_descriptors={
                "color": {}
            }
    )

def hog_pipeline():
    #Solo textura - HOG
    get_map_at_several_ks(
        query_dir=os.path.join("..","queries","qsd1_w4"),
        ks=[1, 5, 10, 20],
        descriptors={"texture_hog": hog_descriptor,
                        },
        descriptors_sim={"texture_hog": hog_similarity,
                            },
        preprocesses=True,
        recompute_bbdd_descriptors=True,
        recompute_query_descriptors=True,
        kwargs_for_descriptors = {
            'texture_hog': {}
        }
    )

def old_pipelines():

    def lbp_pipeline():
        #Solo textura - LBP
        get_map_at_several_ks(
            query_dir=os.path.join("..","old_queries","qsd1_w3"),
            ks=[1, 5, 10, 20],
            descriptors={"texture_lbp": lbp_descriptor,
                         },
            descriptors_sim={"texture_lbp": lbp_similarity,
                             },
            preprocesses=None,
            no_bg=True,
            recompute_bbdd_descriptors=True,
            recompute_query_descriptors=True,
            kwargs_for_descriptors = {
                'texture_lbp': dict(n_blocks=1, P=8, R=10, histogram_size=[255])
            }
        )

    def lbp_and_text_pipeline():

        get_map_at_several_ks(
            query_dir=os.path.join("..","old_queries","qsd1_w3"),
            ks=[1, 5, 10, 20],
            descriptors={"texture_lbp": lbp_descriptor, "text": text_descriptor
                         },
            descriptors_sim={"texture_lbp": lbp_similarity, "text": text_similarity
                             },
            preprocesses=None,
            no_bg=True,
            recompute_bbdd_descriptors=True,
            recompute_query_descriptors=True,
            kwargs_for_descriptors = {
                'texture_lbp': dict(n_blocks=2, P=4, R=3, histogram_size=[255])
            }
        )

    def lbp_and_text_and_color_pipeline():

        get_map_at_several_ks(
            query_dir=os.path.join("..","old_queries","qsd1_w3"),
            ks=[1, 5, 10, 20],
            descriptors={"texture_lbp": lbp_descriptor, "text": text_descriptor, "color": best_color_descriptor
                         },
            descriptors_sim={"texture_lbp": lbp_similarity, "text": text_similarity, "color": generic_histogram_similarity
                             },
            preprocesses=None,
            no_bg=True,
            recompute_bbdd_descriptors=True,
            recompute_query_descriptors=True,
            kwargs_for_descriptors = {
                'texture_lbp': dict(n_blocks=2, P=4, R=3, histogram_size=[255])
            }
        )

    def lbp_and_color_pipeline():

        get_map_at_several_ks(
            query_dir=os.path.join("..","old_queries","qsd1_w3"),
            ks=[1, 5, 10, 20],
            descriptors={"texture_lbp": lbp_descriptor,  "color": best_color_descriptor
                         },
            descriptors_sim={"texture_lbp": lbp_similarity, "color": generic_histogram_similarity
                             },
            preprocesses=None,
            no_bg=True,
            recompute_bbdd_descriptors=True,
            recompute_query_descriptors=True,
            kwargs_for_descriptors = {
                'texture_lbp': dict(n_blocks=2, P=4, R=3, histogram_size=[255])
            }
        )

    def text_pipeline():
        #Solo texto
        get_map_at_several_ks(
                query_dir=os.path.join("..", "old_queries", "qsd1_w2"),
                ks=[1, 5, 10],
                descriptors={"text": text_descriptor},
                descriptors_sim={"text": text_similarity},
                preprocesses=None,
                no_bg=True,
                recompute_bbdd_descriptors=True,
                recompute_query_descriptors=True
        )

    def color_and_text_pipeline():
        get_map_at_several_ks(
                query_dir=os.path.join("..", "old_queries", "qsd2_w3"),
                ks=[1, 5, 10],
                descriptors={"color": best_color_descriptor,
                             "text" : text_descriptor
                             },
                descriptors_sim={"color": generic_histogram_similarity,
                                 "text" : text_similarity
                                 },
                preprocesses=True,
                recompute_bbdd_descriptors=False,
                recompute_query_descriptors=True,
        )

def keypoints_pipeline():
    # Solo keypoints
    get_map_at_several_ks(
            query_dir=os.path.join("..", "queries", "qsd1_w4"),
            ks=[1, 5, 10],
            descriptors={"keypoints": keypoints_descriptor,
                         },
            descriptors_sim={"keypoints": keypoints_similarity,
                             },
            preprocesses=True,
            recompute_bbdd_descriptors=True,
            recompute_query_descriptors=True,
            kwargs_for_descriptors={},
            similarity_threshold=5
    )

if __name__ == "__main__":
    MIN_DIST_TO_BE_MATCH = 90
    hog_pipeline()
    #color_and_text_pipeline()
    #lbp_and_color_pipeline()