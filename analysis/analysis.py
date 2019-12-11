import utilities as utl


def plot_ridf_fields_from_directories():
    images = utl.load_images('../routes/ftl_9_route/', True)
    print('Loaded {} images with shape {}'.format(len(images), images[0].shape))

    # RIDF field plot
    logs = utl.ridf_field(images, 360, 1)
    print(logs.shape)
    utl.plot_3d(logs, show=False, rows_cols_idx=121)


    images = utl.load_images('../routes/ftl_9_route_white_noise/', True)
    print('Loaded {} images with shape {}'.format(len(images), images[0].shape))

    # RIDF field plot
    logs = utl.ridf_field(images, 360, 1)
    print(logs.shape)
    utl.plot_3d(logs, rows_cols_idx=122)


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
# plot_ridf_fields_from_directories()
# plot_cor_coef_v_ridf_from_directory()

