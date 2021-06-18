import utilities as utl


imgs_right = utl.load_images('../test_runs/plants_right2', True)
imgs_front = utl.load_images('../test_runs/plants_forward', True)


imgs_right = utl.crop(imgs_right, y=30, h=15)
imgs_front = utl.crop(imgs_front, y=30, h=15)

utl.display_image(imgs_right[0])
utl.display_image(imgs_front[0])


logs = utl.tridf(imgs_right[0], imgs_front)
utl.plot_multiline(logs)

logs = utl.tridf(imgs_front[0], imgs_right)
utl.plot_multiline(logs)

rotated = utl.rotate(90, imgs_front[0])
utl.heatmap(imgs_right[0], rotated)