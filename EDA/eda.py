import os
import subprocess

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
colorpal = sns.color_palette("husl", 9)

import warnings

DATA_DIR = '../data/input/'
OUTPUT_DIR = 'output/'

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tr_labels = pd.read_csv(DATA_DIR + 'train_labels.csv')
    img_labels = pd.read_csv(DATA_DIR + 'image_labels.csv')
    ss = pd.read_csv(DATA_DIR + 'sample_submission.csv')

    tr_tracking = pd.read_csv(DATA_DIR + 'train_player_tracking.csv')
    te_tracking = pd.read_csv(DATA_DIR + 'test_player_tracking.csv')

    print(tr_labels.head())
    print(tr_labels.columns)


    # Distribution of frames per video file
    play_frame_count = tr_labels[['gameKey','playID','frame']] \
        .drop_duplicates()[['gameKey','playID']] \
        .value_counts()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(play_frame_count, bins=15)
    ax.set_title('Distribution of frames per video file')
    plt.savefig(OUTPUT_DIR+"play_frame_count.png")

    # Distribution bounding box sizes
    tr_labels['area'] = tr_labels['width'] * tr_labels['height']
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(tr_labels['area'].value_counts(),
                bins=10,
                color=colorpal[1])
    ax.set_title('Distribution bounding box sizes')
    plt.savefig(OUTPUT_DIR+'distribution_bounding_box_size.png')


    # Impact type by frame
    plt.figure()
    for i, d in tr_labels.groupby('impactType'):
        if len(d) < 10:
            continue
        d['frame'].plot(kind='kde', alpha=1, figsize=(12, 4), label=i,
                        title='Impact Type by Frame')
        plt.legend()
        plt.savefig(OUTPUT_DIR+'impact_type_by_frame.png')


    # Impact occurance
    pct_impact_occurance = tr_labels[['video','impact']] \
        .fillna(0)['impact'].mean() * 100
    print(f'Of all bounding boxes, {pct_impact_occurance:0.4f}% of them involve an impact event')

    tr_labels[['video','impact','frame']] \
        .fillna(0) \
        .groupby(['frame']).mean() \
        .plot(figsize=(12, 5), title='Occurance of impacts by frame in video.',
            color=colorpal[6])
    plt.saving(OUTPUT_DIR+'Occurance_of_impacts_by_frame_in_videl.png')

    # Pairplot of Bounding Box, Impact vs Non-Impact
    sns.pairplot(tr_labels[['frame','area',
                            'left','width',
                            'top','height',
                            'impact']] \
                    .sample(5000).fillna(0),
                hue='impact')
    plt.saving(OUTPUT_DIR+'Pirplot_of_bbox_vs_impact.png')
