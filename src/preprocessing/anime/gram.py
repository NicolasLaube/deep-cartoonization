import torch

_rgb_to_yuv_kernel = torch.tensor(
    [
        [0.299, -0.14714119, 0.61497538],
        [0.587, -0.28886916, -0.51496512],
        [0.114, 0.43601035, -0.10001026],
    ]
).float()


def gram(input_matrix):
    """
    Calculate Gram Matrix
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    """
    bacth, channels, width, height = input_matrix.size()

    x = input_matrix.view(bacth * channels, width * height)

    gram_matrix = torch.mm(x, x.T)

    # normalize by total elements
    return gram_matrix.div(bacth * channels * width * height)


def rgb_to_yuv(image):
    """
    https://en.wikipedia.org/wiki/YUV
    output: Image of shape (H, W, C) (channel last)
    """
    # -1 1 -> 0 1
    image = (image + 1.0) / 2.0

    yuv_img = torch.tensordot(image, _rgb_to_yuv_kernel, dims=([image.ndim - 3], [0]))

    return yuv_img
