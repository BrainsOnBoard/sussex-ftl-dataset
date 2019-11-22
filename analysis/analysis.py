import utilities as utl


images = utl.load_images('../routes/ftl_9_route/')
print('Loaded {} images with shape {}'.format(len(images), images[0].shape))

# IDF in translation
idfs = utl.imgs_diff(images[0], images)
utl.plot_line(idfs, 'translation idf')

# Correlation coefficient
cors = utl.imgs_coef(images[0], images)
utl.plot_line(cors, 'Correlation coefficients')




