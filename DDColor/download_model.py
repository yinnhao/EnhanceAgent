from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('damo/cv_ddcolor_image-colorization', cache_dir='./modelscope')
print('model assets saved to %s' % model_dir)