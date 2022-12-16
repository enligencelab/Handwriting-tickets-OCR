import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def get_cluster_labels(cells_df):
    n_tables = cells_df['table_id'].max() + 1
    # TODO: Elbow method
    min_pts = int(n_tables * 0.4)
    positions = pd.DataFrame({
        'relative_x': cells_df['x'] / cells_df['table_w'],
        'relative_y': cells_df['y'] / cells_df['table_h']
    })
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
    return img
