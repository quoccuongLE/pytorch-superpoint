import torch
import time


def reshape_pixels2superpixels_legacy(x: torch.Tensor, superpixel_size: int):
    """Transform tensor pixel-wise to
    [batch_size, 1, H, W] -> [batch_size, 65, Hc, Wc]

    Args:
        x (torch.Tensor): pixel-level tensor
        superpixel_size (int): Superpixel size

    Returns:
        torch.Tensor: Reshaped tensor to superpixel level
    """
    assert x.dim() == 4, "Only accept 4-dim tensor!"
    output = x.permute(0, 2, 3, 1)  # [batch_size, 1, H, W] (pytorch style) -> [batch_size, H, W, 1] (tensorflow style)
    (batch_size, s_height, s_width, s_depth) = output.size()
    d_depth = s_depth * superpixel_size**2
    d_width = int(s_width / superpixel_size)
    d_height = int(s_height / superpixel_size)

    t_1 = output.split(superpixel_size, 2)
    stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
    output = torch.stack(stack, 1)
    return output.permute(0, 3, 2, 1)


def reshape_pixels2superpixels(x: torch.Tensor, superpixel_size: int):
    """Transform tensor pixel-wise to
    [batch_size, 1, H, W] -> [batch_size, superpixel_size**2, Hc, Wc]

    Args:
        x (torch.Tensor): pixel-level tensor
        superpixel_size (int): Superpixel size

    Returns:
        torch.Tensor: Reshaped tensor to superpixel level
    """
    assert x.dim() == 4, "Only accept 4-dim tensor!"
    (batch_size, s_depth, s_height, s_width) = x.size()
    assert s_depth == 1, "Only accept input depth = 1"
    d_depth = s_depth * superpixel_size**2
    d_width = int(s_width / superpixel_size)
    d_height = int(s_height / superpixel_size)

    return (
        x.permute(0, 2, 3, 1)  # [batch_size, 1, H, W] -> [batch_size, H, W, 1]
        .reshape(batch_size, s_height, d_width, 8)  # [batch_size, H, w, 8]
        .permute(0, 1, 3, 2)  # [batch_size, H, w, 8] -> [batch_size, H, 8, w]
        .reshape(batch_size, d_height, d_depth, d_width)  # [batch_size, H, 8, w] -> [batch_size, h, 8*8, w]
        .permute(0, 2, 1, 3)  # [batch_size, h, 8*8, w] -> [batch_size, 8*8, h, w]
    )


def reshape_superpixels2pixels_legacy(x: torch.Tensor, superpixel_size: int):
    """Transform tensor superpixel-wise to
    [batch_size, superpixel_size**2, Hc, Wc] -> [batch_size, 1, H, W]

    Args:
        x (torch.Tensor): superpixel-level tensor
        superpixel_size (int): Superpixel size

    Returns:
        torch.Tensor: Reshaped tensor to pixel level
    """
    assert x.dim() == 4, "Only accept 4-dim tensor!"
    assert (
        x.shape[1] == superpixel_size**2
    ), f"Flattened dim {x.shape[1]} must be equal to superpixel_size **2 {superpixel_size**2}"
    output = x.permute(0, 2, 3, 1)  # [batch_size, Hc, Wc, 64]
    (batch_size, d_height, d_width, d_depth) = output.size()
    s_depth = int(d_depth / superpixel_size**2)
    s_width = int(d_width * superpixel_size)
    s_height = int(d_height * superpixel_size)
    # [batch_size, Hc, Wc, 64, 1]
    t_1 = output.reshape(batch_size, d_height, d_width, superpixel_size**2, s_depth)
    # 8 x [batch_size, Hc, Wc, 8, 1]
    spl = t_1.split(superpixel_size, 3)
    # 8 x [batch_size, Hc, W, 1]
    stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
    output = (
        torch.stack(stack, 0)  # [8, batch_size, Hc, W, 1]
        .transpose(0, 1)  # [batch_size, 8, Hc, W, 1]
        .permute(0, 2, 1, 3, 4)  # [batch_size, Hc, 8, W, 1]
        .reshape(batch_size, s_height, s_width, s_depth)  # [batch_size, H, W, 1]
    )
    output = output.permute(0, 3, 1, 2)  # [batch_size, 1, H, W]
    return output


def reshape_superpixels2pixels(x: torch.Tensor, superpixel_size: int):
    """Transform tensor superpixel-wise to
    [batch_size, superpixel_size**2, Hc, Wc] -> [batch_size, 1, H, W]

    Args:
        x (torch.Tensor): superpixel-level tensor
        superpixel_size (int): Superpixel size

    Returns:
        torch.Tensor: Reshaped tensor to pixel level
    """
    assert x.dim() == 4, "Only accept 4-dim tensor!"
    assert (
        x.shape[1] == superpixel_size**2
    ), f"Flattened dim {x.shape[1]} must be equal to superpixel_size **2 {superpixel_size**2}"
    (batch_size, d_depth, d_height, d_width) = x.size()
    s_depth = int(d_depth / superpixel_size**2)
    s_height = int(d_height * superpixel_size)
    return (
        x.permute(0, 3, 2, 1)  # [batch_size, Wc, Hc, 64]
        .reshape(batch_size, d_width, -1, superpixel_size)  # [batch_size, Wc, H, 8]
        .permute(0, 1, 3, 2)  # [batch_size, Wc, 8, H]
        .reshape(batch_size, s_depth, -1, s_height)  # [batch_size, 1, W, H]
        .permute(0, 1, 3, 2)  # [batch_size, 1, H, W]
    )


if __name__ == "__main__":
    # [batch_size, 1, H, W] -> [batch_size, 65, Hc, Wc]
    H = 120
    W = 160
    x = torch.arange(H*W).reshape(1, 1, H, W)
    sp_size = 8
    start = time.time()
    for _ in range(100):
        y1 = reshape_pixels2superpixels_legacy(x, sp_size)
    print(time.time() - start)

    start = time.time()
    for _ in range(100):
        y2 = reshape_pixels2superpixels(x, sp_size)
    print(time.time() - start)
    assert torch.equal(y1, y2)

    y = reshape_pixels2superpixels(x, sp_size)

    start = time.time()
    for _ in range(100):
        x1 = reshape_superpixels2pixels_legacy(y, sp_size)
    print(time.time() - start)

    start = time.time()
    for _ in range(100):
        x2 = reshape_superpixels2pixels(y, sp_size)
    print(time.time() - start)
    assert torch.equal(x1, x2)
