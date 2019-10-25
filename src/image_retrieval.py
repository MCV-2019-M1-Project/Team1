from import_manager import (
    import_bbdd_paintings,
    import_query_paintings,
    import_gt_corresps
)

import pickle
import os
import cv2.cv2 as cv2
import ml_metrics as metrics
import datetime
import random
import numpy as np

#0: Implementar retrieval system #DONE

#TODO #0.1: Testear y debugar retrieval system

#TODO #1: Task 2: Implementar el image retrieval SOLO con text recognition (QSD1) map@1: - map@5: - map@10:

#TODO #2: Task 2: Implementar el image retrieval SOLO con color descriptor (QSD1) map@1: - map@5: - map@10:

#TODO #3: Task 3: Implementar el image retrieval SOLO con texture descriptor (QSD1) map@1: - map@5: - map@10:

#TODO #4: Task 4: Combinar descriptores anteriores - polling system (QSD1) map@1: - map@5: - map@10:

#TODO #5: Task 5: Retrieval con QSD2: Integrar la retirada del fondo y soportar 2 cuadros por query. (QSD2) map@1: - map@5: - map@10:

#TODO #6: Submission (María Rodríguez): Para cada imagen de las queries de test (QST1 y QST2) escribir un file con el texto extraído


def how_similar(desc1, desc2):
    return random.random()

def apply_preprocessing(image_filepath, preprocessing, no_bg, single_sure):
    im_source = cv2.imread(image_filepath)
    for f in preprocessing:
        im_dest = f(im_source)
        im_source = im_dest.copy()
    return im_source



### DE AQUÍ PARA ABAJO IS DONE ### STILL NOT TESTED ###

def merge_similarities(sim_by_desc):
    """
    :param sim_by_desc: { desc_name: [a, b, c], ... } being a = similarity of bbdd_item 0, b =  = similarity of bbdd_item 1, ...
    :return: the final merged order: [index_at_bbdd_top1_similar, index_at_bbdd_top2_similar, ...]
    """

    def normalize(v):
        "escala los valores de v para ponerlos entre 0 y 1"
        return list((v - np.min(v))/np.ptp(v))


    scaled_sim_by_desc = {k: normalize(sim_by_desc[k]) for k in sim_by_desc.keys()}
    merged_similarity = [0.0] * len(sim_by_desc.values()[0])
    for sim_vec in scaled_sim_by_desc.values():
        for i, sim in enumerate(sim_vec):
            merged_similarity[i] += sim

    final_order = [t[0] for t in sorted(enumerate(merged_similarity), key=lambda x: x[1], reverse=True)]
    return final_order

def order_by_similarity(bbdd_descriptors, query_descriptor, descriptors_sim):
    """
    :param bbdd_descriptors: [dict(im_num -> dict(desc_name -> desc_value)), ... ]
    :param query_descriptor: dict(desc_name -> desc_value)
    :param descriptors_sim: { desc_name: fun(obj, obj) -> sim , ...}
    :return: indexes of the bbdd ordered by similarity
    """
    # Para cada descriptor un orden. Despues, merge.
    sim_by_desc = {}

    for desc_name, desc_obj in query_descriptor.items():
        sim_list = [descriptors_sim[desc_name](desc_obj, bbdd_descriptors[i][desc_name])
                    for i in sorted(bbdd_descriptors.keys())]
        sim_by_desc[desc_name] = sim_list[:]

    final_order = merge_similarities(sim_by_desc)
    return final_order

def compute_similarity_all_queryset(bbdd_descriptors, query_descriptors, descriptors_sim, k_closest=10):
    """
    :param query_descriptors: [im_num: { desc_name: fun(I) -> obj , ... }, ...]
    :param bbdd_descriptors:  [im_num: { desc_name: fun(I) -> obj , ... }, ...]
    :param descriptors_sim: { desc_name: fun(obj, obj) -> sim , ...}

    """
    predictions = []
    for query_key in sorted(query_descriptors.keys()): # Itero sobre todas las queries
        descriptors_list = query_descriptors[query_key] # Lista que contiene un diccionario de descriptores por cada cuadro encontrado
        query_image_result = []
        for query_descriptor in descriptors_list: # query_descriptor es un { desc_name: fun(I) -> obj , ... }
            query_image_result.append(
                    order_by_similarity(bbdd_descriptors,
                                        query_descriptor,
                                        descriptors_sim)[:k_closest]
            )
        predictions.append(query_image_result[:])
    return predictions

def get_descriptors_given_query_dir(descriptors_definition,
                                    folder,
                                    recompute=True,
                                    preprocessing=None,
                                    no_bg=False,
                                    single_sure=False):
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
    if not recompute:
        try:
            with open(os.path.join(folder,"descriptors.pkl"), 'rb') as handle:
                ds = pickle.load(handle)
                return ds
        except FileNotFoundError:
            pass


    folder_images_paths = [os.path.join(folder, image_filename)
                          for image_filename in
                          sorted(filter(
                                  lambda f: f.endswith('.jpg'), os.listdir(folder)))]

    # Para cada número de imagen, una lista que acabará conteniendo un diccionario por cada cuadro contenido en la imagen
    descriptors = {k: [{}, {}] for k in [int(os.path.split(x)[-1]) for x in folder_images_paths]}

    for image_filepath in folder_images_paths:
        im_num = int(os.path.split(image_filepath)[-1])
        for descriptor_name, descriptor_func in descriptors_definition.items():
            #try:
            if preprocessing is not None:
                list_containing_one_or_two_images = apply_preprocessing(image_filepath,
                                                                        preprocessing,
                                                                        no_bg,
                                                                        single_sure)
            else:
                list_containing_one_or_two_images = [cv2.imread(image_filepath)]

            for i, im in enumerate(list_containing_one_or_two_images):
                descriptors[im_num][i][descriptor_name] = descriptor_func(im)
            #except BaseException as e:
            #    print(e)

        # Cleaning empty dicts for that images that contained only one painting
    descriptors = {k: [x for x in descriptors[k] if len(x) > 0] for k in descriptors.keys()}
    return descriptors

def get_map_at_several_ks(query_dir, ks,
                          descriptors,
                          descriptors_sim,
                          preprocesses,
                          no_bg=False,
                          single_sure=False,
                          submission=False,
                          recompute_bbdd_descriptors=False,
                          recompute_query_descriptors=False):
    """
    :param query_dir: directory containing the query images

    :param ks: list containing all the ks for computing the map@k: Example: [1,5,10]

    :param descriptors: { desc_name: fun(I) -> obj , ... }

    :param descriptors_sim: { desc_name: fun(obj, obj) -> sim , ...}

    :param preprocesses: list of functions that take an image and return an image

    :param no_bg: boolean that says if images need de-backgrounding

    :param single_sure: boolean that says if the dataset contain only single images

    :param submission: boolean that tells if the execution is for submission (no GT)

    :param recompute_bbdd_descriptors: tells if descriptors of the bbdd must be recomputed

    :return: Diccionario con las K y map@k. Ejemplo: dict(1: 0.6, 3:0.8, 10:0.95)
    """

    try:
        bbdd_descriptors = get_descriptors_given_query_dir(descriptors,
                                                           os.path.join("..","bbdd"),
                                                           recompute_bbdd_descriptors)
    except Exception as e:
        print(" **************** Exception ocurred while getting the bbdd descriptors **************** ")
        print(e)
        raise e

    else:
        if recompute_bbdd_descriptors:
            with open(os.path.join("..","bbdd","descriptors.pkl"), "w") as f:
                pickle.dump(bbdd_descriptors, f)

    try:
        query_descriptors = get_descriptors_given_query_dir(descriptors,
                                                            query_dir,
                                                            recompute_query_descriptors,
                                                            preprocessing=preprocesses,
                                                            no_bg=no_bg,
                                                            single_sure=single_sure)
    except Exception as e:
        print("**************** Exception ocurred while getting the query descriptors **************** ")
        print(e)
        raise e
    else:
        if recompute_query_descriptors:
            with open(os.path.join(query_dir,"descriptors.pkl"), "w") as f:
                pickle.dump(query_descriptors, f)

    predictions = compute_similarity_all_queryset(bbdd_descriptors, query_descriptors, descriptors_sim, k_closest=10)

    if submission:
        with open(os.path.join(query_dir, "result.pkl"), "w") as f:
            pickle.dump(query_descriptors, f)

    if not submission:
        with open(os.path.join(query_dir,"gt_corresps.pkl") ,'rb') as handle:
            gt = pickle.load(handle)

        map_ = {k: 0.0 for k in ks}
        number_of_paintings_queried = 0
        for image_prediction_list, gt_list in zip(predictions, gt):
            if len(image_prediction_list) == len(gt_list): # El numero de cuadros detectados es el bueno

                for single_prediction, single_gt in zip(image_prediction_list,gt_list):
                    for k in ks:
                        map_[k] += metrics.mapk(single_gt, single_prediction, k)
                    number_of_paintings_queried += 1

            elif len(image_prediction_list) < len(gt_list): # Habia 2 cuadros y solo hemos detectado uno
                number_of_paintings_queried += 1

            elif len(image_prediction_list) > len(gt_list): # Habia 1 cuadro y hemos detectado 2
                number_of_paintings_queried += 1

        with open(os.path.join(query_dir, "map_at_k"+datetime.datetime.now().isoformat()+".txt"), "w") as f:
            for k, m in sorted(map_.items()):
                f.write("map@"+str(k)+": "+"%.3f" % m+"\n")
                print("map@"+str(k)+": "+"%.3f" % m+"\n")








