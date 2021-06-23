import utilities as utl

img_set6 = utl.load_images('../test_runs/white_walls/ftl_6_white_walls/', True)
print('Loaded {} images with shape {}'.format(len(img_set6), img_set6[0].shape))

img_set9 = utl.load_images('../test_runs/white_walls/ftl_9_white_walls/', True)
print('Loaded {} images with shape {}'.format(len(img_set6), img_set6[0].shape))

set6_img1 = img_set6[0]
set9_img1 = img_set9[0]
print(set6_img1.shape)
utl.display_image(set6_img1)
utl.display_image(set9_img1)


ridfs = []
ridfs.append(utl.ridf(set6_img1, set6_img1, 360, 1))
ridfs.append(utl.ridf(set6_img1, set9_img1, 360, 1))

utl.plot_multiline(ridfs, ['set6vset6', 'set6vset9'])


def plot_ridf_fields_from_directories():
    images = utl.load_images('../test_runs/white_walls/ftl_6_white_walls/', True)
    print('Loaded {} images with shape {}'.format(len(images), images[0].shape))

    # RIDF field plot
    logs = utl.tridf(images[0], images, 360, 1)
    print(logs.shape)
    utl.plot_3d(logs, show=False, rows_cols_idx=121)


    images = utl.load_images('../test_runs/white_walls/ftl_9_white_walls/', True)
    print('Loaded {} images with shape {}'.format(len(images), images[0].shape))

    # RIDF field plot
    logs = utl.tridf(images[0], images, 360, 1)
    print(logs.shape)
    utl.plot_3d(logs, rows_cols_idx=122)

plot_ridf_fields_from_directories()

