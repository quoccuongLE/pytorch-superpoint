import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.model_utils import SuperPointNet_process
from models.model_wrap import PointTracker
from utils.draw import draw_keypoints
from utils.loader import dataLoader_test as dataLoader
from utils.loader import get_module
from utils.print_tool import datasize, print_dict_attr
from utils.utils import toNumpy


def draw_matches(
    rgb1,
    rgb2,
    match_pairs,
    lw=0.5,
    color="g",
    if_fig=True,
    filename="matches.png",
    show=False,
):
    plt.clf()
    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=rgb1.dtype)
    canvas[:h1, :w1] = rgb1[:, :, np.newaxis]
    canvas[:h2, w1:] = rgb2[:, :, np.newaxis]
    if if_fig:
        fig = plt.figure(figsize=(15, 5))
    plt.axis("off")
    plt.imshow(canvas, zorder=1)

    xs = match_pairs[:, [0, 2]]
    xs[:, 1] += w1
    ys = match_pairs[:, [1, 3]]

    alpha = 1
    sf = 5
    # lw = 0.5
    # markersize = 1
    markersize = 2

    plt.plot(
        xs.T,
        ys.T,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        marker="o",
        markersize=markersize,
        fillstyle="none",
        color=color,
        zorder=2,
        # color=[0.0, 0.8, 0.0],
    )
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    print("#Matches = {}".format(len(match_pairs)))
    if show:
        plt.show()


def main():
    filename = "configs/magicpoint_repeatability_heatmap_synth.yaml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_dtype(torch.float32)
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    # data loading

    task = config["data"]["dataset"]
    data = dataLoader(config, dataset=task)
    test_set, test_loader = data["test_set"], data["test_loader"]

    datasize(test_loader, config, tag="test")

    # model loading

    Val_model_heatmap = get_module("", config["front_end_model"])

    ## load pretrained
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    net = val_agent.net
    # load data
    for i, sample in tqdm(enumerate(test_loader)):
        img_0, img_1 = sample["image"], sample["warped_image"]
        if i > 1:
            break

    img_pair = torch.cat((img_0, img_1), dim=0)
    outs = net(img_pair.to(device))
    print("outs: ", list(outs))
    # process outputs
    print_dict_attr(outs, "shape")
    params = {
        "out_num_points": 500,
        "patch_size": 5,
        "device": device,
        "nms_dist": 4,
        "conf_thresh": 0.015,
    }
    # sp_processer = SuperPointNet_process(**params)
    # outs_post = net.process_output(sp_processer)
    outs_post = net.post_process(outs)

    print("outs: ", list(outs_post))

    print_dict_attr(outs_post, "shape")
    outs["semi"] = torch.nan_to_num(outs["semi"], nan=0.0)
    # outs_post = net.process_output(sp_processer)
    outs_post = net.post_process(outs)

    pts_int = outs_post["pts_int"]
    pts_offset = outs_post["pts_offset"]
    pts_desc = outs_post["pts_desc"]

    for i in range(2):
        img = draw_keypoints(
            toNumpy(img_pair[i].squeeze()),
            toNumpy((pts_int[i] + pts_offset[i]).squeeze()).transpose(),
        )
        plt.imshow(img)
        plt.savefig(f"tmp/keypoints_{i}_synth.png")

    # tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    # for i in range(2):
    #     f = lambda x: toNumpy(x.squeeze())
    #     tracker.update(f(pts_int[i]).transpose(), f(pts_desc[i]).transpose())

    # matches = tracker.get_matches().T
    # print("matches: ", matches.transpose().shape)
    # draw_matches(f(img_pair[0]), f(img_pair[1]), matches, filename="tmp/test.png")

if __name__ == "__main__":
    main()
