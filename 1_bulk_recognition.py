import os
from itertools import count
from tickets.table_layout import pdf_to_image, locate_table_rotated, cells_position
import pandas as pd
import pickle
from tickets.cell_correspondence import get_cluster_labels, group_cells, visualize_anchors
from lstm_rnn_ctc.table_ocr import table_to_text
from tqdm import tqdm

# %% file I/O
filename = 'data/style1.pdf'
name = os.path.splitext(os.path.split(filename)[1])[0]

# %% table layout
pages_generator = pdf_to_image(filename)
page_number_generator = count()
cells = []
print('[1/3] Analyze layout of the tables.')
with open(f'raw/{name}_tables.pkl', 'wb') as f:
    for page, channel in pages_generator:
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

# %% cell correspondence
print('[2/3] Correspond cells detected in each image.')
cells_labeled = get_cluster_labels(cells)
anchors = group_cells(cells_labeled)
visualize_anchors(table, anchors, f'results/{name}_annotated_anchors.jpg')

# %% table OCR
print('[3/3] Perform handwriting Chinese OCR of the tables.')
text = []
with open(f'raw/{name}_tables.pkl', 'rb') as f:
    for i in tqdm(range(page_number + 1)):
        table = pickle.load(f)
        text.append(table_to_text(table, anchors))
text = pd.DataFrame(columns=anchors['label'].values, data=text)
text.to_excel(f'results/{name}_text.xlsx', index=False)
