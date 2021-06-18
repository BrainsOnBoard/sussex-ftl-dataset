import utilities as utl

# images = utl.load_images('../test_runs/background_test3', True)
#
#
# images = utl.crop(images, y=30, h=15)
# utl.display_image(images[60])
#
# rotated = utl.rotate(90, images[60])
# utl.heatmap(images[60], rotated)
#
# # RIDF field plot
# logs = utl.ridf_field(images[58:62], 360, 1)
# print(logs.shape)
# utl.plot_multiline(logs)
# # utl.plot_3d(logs)
#
# logs = utl.tridf(images[60], images[40:80], 360, 1)
# print(logs.shape)
# utl.plot_3d(logs)




def plot_ridf_fields_from_directories():
    images = utl.load_images('../test_runs/background_test2', True)

    # RIDF field plot
    logs = utl.tridf(images[60], images, 360, 1)
    print(logs.shape)
    utl.plot_3d(logs, show=False, rows_cols_idx=121, title='empty')


    images = utl.load_images('../test_runs/background_test3', True)

    # RIDF field plot
    logs = utl.tridf(images[90], images, 360, 1)
    print(logs.shape)
    utl.plot_3d(logs, rows_cols_idx=122, title='with plants')


def plot_cor_coef_v_ridf_from_directory():
    # Change to false to lod images with channels (RGB)
    images = utl.load_images('../routes/ftl_9_route/', True)
    print('Loaded {} images with shape {}'.format(len(images), images[0].shape))

    # Correlation coefficient field plot
    logs = utl.cor_coef_field(images, 360, 1)
    print(logs.shape)
    utl.plot_3d(logs, show=False, rows_cols_idx=121)

    # Translational RIDF
    logs = utl.tridf(images[0], images, 360, 1)
    print(logs.shape)
    utl.plot_3d(logs, rows_cols_idx=122)


# # Uncomment to execute the functions
plot_ridf_fields_from_directories()
# plot_cor_coef_v_ridf_from_directory()



