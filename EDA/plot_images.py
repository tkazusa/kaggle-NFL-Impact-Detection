import os

import cv2
import matplotlib.patches as patches
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
colorpal = sns.color_palette("husl", 9)

DATA_DIR = '../data/input/'
OUTPUT_DIR = 'output/'


def plot_example_image(dir_name, img_fn, ax, highlight_color='r', highlight_alpha=0.5):
    img_data = cv2.imread(dir_name + img_fn)
    print(img_data)
    ax.imshow(img_data)
    ax.grid(False)

    # Create a Rectangle patch
    for i, d in img_labels.loc[img_labels['image'] == img_fn].iterrows():

        rect = patches.Rectangle((d['left'],
                                  d['top']),
                                  d['width'],
                                 d['height'],
                                 linewidth=1,
                                 edgecolor=highlight_color,
                                 facecolor=highlight_color,
                                alpha=highlight_alpha)
        ax.add_patch(rect)
    ax.axis('off')
    ax.set_title(img_fn)
    return ax

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tr_labels = pd.read_csv(DATA_DIR + 'train_labels.csv')
    img_labels = pd.read_csv(DATA_DIR + 'image_labels.csv')
    ss = pd.read_csv(DATA_DIR + 'sample_submission.csv')

    tr_tracking = pd.read_csv(DATA_DIR + 'train_player_tracking.csv')
    te_tracking = pd.read_csv(DATA_DIR + 'test_player_tracking.csv')    

    # Loop through 8 example images
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))
    axs = axs.flatten()
    i = 0
    for example_image in img_labels.sample(8, random_state=999)['image']:
        plot_example_image(DATA_DIR, example_image, axs[i])
        i += 1
    plt.savefig(OUTPUT_DIR+'images.png')