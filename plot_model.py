from keras.models import load_model
from keras.utils.vis_utils import plot_model
import caption_generator_resnet


cg = caption_generator_resnet.CaptionGenerator()
model_resnet=cg.create_model(ret_model=True)
#model_resnet.load_weights('weights-improvement-resnet-18.hdf5')
plot_model(model_resnet,'resnet_latest.png',show_shapes=True)
