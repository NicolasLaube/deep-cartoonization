import torch


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
