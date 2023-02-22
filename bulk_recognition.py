import os
import pickle
from itertools import count
import cv2
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from lstm_rnn_ctc.table_ocr import table_to_text


def get_cluster_labels(cells_df):
    positions = pd.DataFrame({
        'relative_x': cells_df['x'] / cells_df['table_w'],
        'relative_y': cells_df['y'] / cells_df['table_h']
    })
    n_tables = cells_df['table_id'].max() + 1
    min_pts = int(n_tables * 0.4)

    # empirical method
    relative_w = cells_df['w'] / cells_df['table_w']
    relative_h = cells_df['h'] / cells_df['table_h']
    eps = np.sqrt(relative_w.min() ** 2 + relative_h.min() ** 2) * 1.22475

    dbscan = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1)
    cells_df['labels'] = dbscan.fit_predict(positions)
    cells_df = cells_df.loc[cells_df['labels'] != -1, :]
    cells_df.drop_duplicates(subset=['table_id', 'labels'])
    return cells_df


def group_cells(cells_df):
    n_tables = cells_df['table_id'].max() + 1
    cells_subtotals = []
    for (label, cells_subset) in cells_df.groupby('labels'):
        cells_subtotals.append({
            'label': label + 1,
            'frequency': cells_subset.shape[0] / n_tables,
            'avg_relative_x': (cells_subset['x'] / cells_subset['table_w']).mean(),
            'avg_relative_y': (cells_subset['y'] / cells_subset['table_h']).mean(),
            'avg_relative_w': (cells_subset['w'] / cells_subset['table_w']).mean(),
            'avg_relative_h': (cells_subset['h'] / cells_subset['table_h']).mean(),
        })
    return pd.DataFrame(cells_subtotals)


def visualize_anchors(img, anchors, output_path):
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img[:, :, np.newaxis], cv2.COLOR_GRAY2RGB)
    h, w, _ = img.shape
    border_width = int(max(min(h / 540, w / 540), 2))
    for _, anchor in anchors.iterrows():
        x1 = int(anchor['avg_relative_x'] * w)
        y1 = int(anchor['avg_relative_y'] * h)
        x2 = x1 + int(anchor['avg_relative_w'] * w)
        y2 = y1 + int(anchor['avg_relative_h'] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), border_width)  # BGR
        # https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
        cv2.putText(img, f"[{int(anchor['label'])}]{int(anchor['frequency'] * 100)}",
                    org=(x1, y2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 255),  # BGR
                    thickness=border_width)
    cv2.imwrite(output_path, img)


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



if __name__ == '__main__':
    # file I/O
    filename = 'data/style1.pdf'
    name = os.path.splitext(os.path.split(filename)[1])[0]

    # table layout
    pages_generator_ = pdf_to_image(filename)
    page_number_generator = count()
    cells = []
    print('[1/3] Analyze layout of the tables.')
    with open(f'raw/{name}_tables.pkl', 'wb') as f:
        for page, channel in pages_generator_:
            page_number = next(page_number_generator)  # counter from 0
            # Flush the same line:
            # use `print()` to start a new line, giving space to the first "cursor up" command, before the loop
            # use the suffix `\x1b[1A\x1b[2K` in each print to move the cursor up and clean the line
            # if the string length monotonically increase in each iteration, we simplify this version as the following
            print(f'\rAnalyzing page {page_number + 1}.', end='')
            table = locate_table_rotated(page[:, :, 0])
            pickle.dump(table, f)
            cells_per_table = cells_position(table, page_number)
            cells.append(cells_per_table)
    print()  # start a new line
    cells = pd.concat(cells)

    # cell correspondence
    print('[2/3] Correspond cells detected in each image.')
    cells_labeled = get_cluster_labels(cells)
    anchors_ = group_cells(cells_labeled)
    visualize_anchors(table, anchors_, f'results/{name}_annotated_anchors.jpg')

    # table OCR
    print('[3/3] Perform handwriting Chinese OCR of the tables.')
    text = []
    with open(f'raw/{name}_tables.pkl', 'rb') as f:
        for i in tqdm(range(page_number + 1)):
            table = pickle.load(f)
            text.append(table_to_text(table, anchors_))
    text = pd.DataFrame(columns=anchors_['label'].values, data=text)
    text.to_excel(f'results/{name}_text.xlsx', index=False)
