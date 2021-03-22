import cairo
import numpy as np
from .utils import cairo_argb_to_opencv_bgr


def _make_cairo_surface(width, height):
    arr = np.zeros((height, width), np.uint32)
    surface = cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_ARGB32, width, height)
    return surface, arr


def put_text(text, fontsize, color, width, height, fontfamily='Droid Sans Mono', bold=False):
    """
    :param text:
    :param fontsize:
    :param color:
    :param width:
    :param height:
    :param fontfamily:
    :return: an OpenCV BGR image.
    """
    surface, canvas = _make_cairo_surface(width, height)
    cr = cairo.Context(surface)
    if bold:
        weight = cairo.FONT_WEIGHT_BOLD
    else:
        weight = cairo.FONT_WEIGHT_NORMAL

    cr.select_font_face(fontfamily, cairo.FONT_SLANT_NORMAL, weight)
    cr.set_font_size(fontsize)

    cr.set_source_rgb(0, 0, 0)
    cr.paint()
    cr.set_source_rgb(color[0], color[1], color[2])
    cr.move_to(0, fontsize)
    cr.show_text(text)

    return cairo_argb_to_opencv_bgr(canvas)
