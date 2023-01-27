from tickets.table_layout import pdf_to_image
import cv2
import numpy as np

# %%
i = 1
# i = 2
filename = f'data/style{i}.pdf'
name = f'style{i}'

# %%
pages_generator = pdf_to_image(filename, page_indices=[0])
page, channel = next(pages_generator)
cv2.imwrite(f'results/textbook/style{i}_page.jpg', page)


# %%
def locate_table(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return int(x), int(y), int(x + w), int(y + h)


def locate_table_rotated(contour):
    box_ = cv2.minAreaRect(contour)
    box_ = cv2.boxPoints(box_)
    box_ = np.int0(box_)
    return box_


img = cv2.bitwise_not(page[:, :, 0])  # conver to black background
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
size = [cv2.contourArea(x) for x in contours]  # Slightly faster than np.vectorize but not significant
largest_outer_contour = contours[np.argmax(size)]
table = cv2.cvtColor(page, cv2.COLOR_GRAY2RGB)
x1, y1, x2, y2 = locate_table(largest_outer_contour)
cv2.rectangle(table, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BGR
box = locate_table_rotated(largest_outer_contour)
cv2.drawContours(table, [box], 0, (0, 255, 0), 2)  # BGR
cv2.imwrite(f'results/textbook/style{i}_table.jpg', table)
