import pypdfium2 as pdfium
import cv2
import numpy as np
import pandas as pd


def pdf_to_image(path, page_indices=None):
    pdf = pdfium.PdfDocument(path)
    if page_indices is None:
        page_indices = range(len(pdf))
    # https://pypdfium2.readthedocs.io/en/stable/python_api.html#pypdfium2._helpers.page.PdfPage.render_base
    pages_generator = pdf.render_to(pdfium.BitmapConv.numpy_ndarray, page_indices=page_indices,
                                    scale=300 / 72, greyscale=True)  # scale unit: 72 dpi
    return pages_generator


def locate_table(img):
    img = cv2.bitwise_not(img)  # conver to black background
    # significantly faster than cv2.RETR_TREE
    # extracting subtree from a hierarchy unsolved -> use cv2.RETR_EXTERNAL instead of cv2.RETR_TREE
    # not analyzing hierarchy inside the table in this function
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size = [cv2.contourArea(x) for x in contours]  # Slightly faster than np.vectorize but not significant
    largest_outer_contour = contours[np.argmax(size)]
    x, y, w, h = cv2.boundingRect(largest_outer_contour)
    return img[y:y + h, x:x + w]


def locate_table_rotated(img):
    img = cv2.bitwise_not(img)  # conver to black background
    # significantly faster than cv2.RETR_TREE
    # extracting subtree from a hierarchy unsolved -> use cv2.RETR_EXTERNAL insteead of cv2.RETR_TREE
    # not analyzing hierarchy inside the table in this function
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size = [cv2.contourArea(x) for x in contours]  # Slightly faster than np.vectorize but not significant
    largest_outer_contour = contours[np.argmax(size)]
    # automatically rotate the table
    # https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
    box = cv2.minAreaRect(largest_outer_contour)
    w, h = box[1]
    source_points = cv2.boxPoints(box)
    if box[2] > 45:  # (45, 90) left is lower and right is higher
        destination_points = np.array([[0, 0], [h, 0], [h, w], [0, w]])
        transformation = cv2.getPerspectiveTransform(source_points.astype('float32'),
                                                     destination_points.astype('float32'))
        rotated_table = cv2.warpPerspective(img, transformation, (int(h), int(w)))
    else:  # (0, 45) left is higher and right is lower
        destination_points = np.array([[0, h], [0, 0], [w, 0], [w, h]])
        transformation = cv2.getPerspectiveTransform(source_points.astype('float32'),
                                                     destination_points.astype('float32'))
        rotated_table = cv2.warpPerspective(img, transformation, (int(w), int(h)))
    return rotated_table


def cells_position(img, table_id):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape
    sq_lb, sq_ub = max(784, h * w * 6e-4), h * w * 0.9
    positions = np.array([cv2.boundingRect(x) for x in contours if sq_lb < cv2.contourArea(x) < sq_ub])
    # when contourArea > sq_lb, cell w*h > sq_lb w.p.1
    positions = positions[positions[:, 2] * positions[:, 3] < sq_ub, :]

    positions_df = pd.DataFrame(columns=['x', 'y', 'w', 'h'], data=positions)
    positions_df['table_id'] = table_id
    positions_df['table_w'] = w
    positions_df['table_h'] = h
    return positions_df
