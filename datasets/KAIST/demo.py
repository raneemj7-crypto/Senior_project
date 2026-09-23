import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')

        # KAIST dataset format: x, y, w, h
        x = int(bndbox.find('x').text)
        y = int(bndbox.find('y').text)
        w = int(bndbox.find('w').text)
        h = int(bndbox.find('h').text)

        # Convert to [x1, y1, x2, y2]
        boxes.append({'name': name, 'bbox': [x, y, x + w, y + h]})
    return boxes

def visualize_sample(vis_img_path, lw_img_path, xml_path):
    # 1. Load images and label
    vis_img = cv2.imread(vis_img_path)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    lw_img = cv2.imread(lw_img_path)
    lw_img = cv2.cvtColor(lw_img, cv2.COLOR_BGR2RGB)

    boxes = parse_xml(xml_path)

    # 2. Set axes
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    titles = ['Visible (RGB)', 'Thermal (LWIR)']
    images = [vis_img, lw_img]

    for ax, img, title in zip(axes, images, titles):
        # Draw boxes
        for obj in boxes:
            box = obj['bbox']
            # Rectangle: (x1, y1), (x2, y2)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img, obj['name'], (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Usage
    xml_file = "annotations-xml-new-sanitized/set04/V001/I00097.xml"
    vis_file = "images/set04/V001/visible/I00097.jpg"
    lw_file = "images/set04/V001/lwir/I00097.jpg"
    visualize_sample(vis_file, lw_file, xml_file)