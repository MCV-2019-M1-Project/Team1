from import_manager import (
    import_bbdd_paintings,
    import_query_paintings,
    import_gt_corresps
)

import pickle
import os


#TODO #0: Implementar retrieval system

#TODO #1: Task 2: Implementar el image retrieval SOLO con text recognition (QSD1) map@1: - map@5: - map@10:

#TODO #2: Task 2: Implementar el image retrieval SOLO con color descriptor (QSD1) map@1: - map@5: - map@10:

#TODO #3: Task 3: Implementar el image retrieval SOLO con texture descriptor (QSD1) map@1: - map@5: - map@10:

#TODO #4: Task 4: Combinar descriptores anteriores - polling system (QSD1) map@1: - map@5: - map@10:

#TODO #5: Task 5: Retrieval con QSD2: Integrar la retirada del fondo y soportar 2 cuadros por query. (QSD2) map@1: - map@5: - map@10:

#TODO #6: Submission (María Rodríguez): Para cada imagen de las queries de test (QST1 y QST2) escribir un file con el texto extraído


def get_bbdd_descriptors(descriptors, recompute_bbdd_descriptors):
    """
    Abstraction layer that either obtains a cached version of the bbdd descriptors or recomputes and stores them
    :param descriptors: list of functions that take an image and return an array
    :param recompute_bbdd_descriptors:
    :return: dict
    """
    if not recompute_bbdd_descriptors:
        try:
            with open(os.path.join("..","data","bbdd_descriptors.pkl"), 'r') as handle:
                ds = pickle.load(handle)
                return ds
        except FileNotFoundError:
            pass





def get_map_at_several_ks(query_dir, ks, descriptors, preprocesses, no_bg=False,
                          single_sure=False, submission=False, recompute_bbdd_descriptors=False):
    """
    :param query_dir: directory containing the query images
    :param ks: list containing all the ks for computing the map@k: Example: [1,5,10]
    :param descriptors: list of functions that take an image and output an array
    :param preprocesses: list of functions that take an image and return an image
    :param no_bg: boolean that says if images need de-backgrounding
    :param single_sure: boolean that says if the dataset contain only single images
    :param submission: boolean that tells if the execution is for submission (no GT)
    :param recompute_bbdd_descriptors: tells if descriptors of the bbdd must be recomputed
    :return: Diccionario con las K y predicciones. Ejemplo: dict(1: 0.6, 3:0.8, 10:0.95), predicted
    """

    query_images_paths = [os.path.join(query_dir, image_filename)
                          for image_filename in
                          sorted(filter(
                                  lambda f: f.endswith('.jpg'), os.listdir(query_dir)))]

    bbdd_descriptors = get_bbdd_descriptors(descriptors)
    if recompute_bbdd_descriptors:
        with open(os.path.join("..","data","bbdd_descriptors.pkl"), "w") as f:
            pickle.dump(bbdd_descriptors, f)


