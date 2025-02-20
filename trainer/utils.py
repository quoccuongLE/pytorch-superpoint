import numpy as np
import torch


def thd_img(img: np.ndarray, threshold: float = 0.015) -> np.ndarray:
    img[img < threshold] = 0
    img[img >= threshold] = 1
    return img


def get_ndarray(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def img_overlap(
    img_r: np.ndarray, img_g: np.ndarray, img_gray: np.ndarray
) -> np.ndarray:  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def update_overlap(
    images_dict, labels_warp_2D, heatmap_nms_batch, img_warp, name
):
    nms_overlap = [
        img_overlap(
            get_ndarray(labels_warp_2D[i]),
            heatmap_nms_batch[i],
            get_ndarray(img_warp[i]),
        )
        for i in range(heatmap_nms_batch.shape[0])
    ]
    nms_overlap = np.stack(nms_overlap, axis=0)
    images_dict.update({name + "_nms_overlap": nms_overlap})


to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)
