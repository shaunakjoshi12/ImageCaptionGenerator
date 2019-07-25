import cPickle as pickle
from keras.preprocessing import image
from resnet101 import resnet101_model
#from vgg16 import VGG16
import numpy as np 
from keras.applications.imagenet_utils import preprocess_input	

def load_image(path):
    img = image.load_img(path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return np.asarray(x)

def load_encoding_model():
	weights_path = 'resnet101_weights_tf.h5'
	return resnet101_model(weights_path)

def get_encoding(model, img):
	image = load_image(img)
	pred = model.predict(image)
	pred = np.reshape(pred, pred.shape[1])
	print "Encoding image: "+str(img)
	print pred.shape
	return pred

def prepare_image_encoding(image_name, no_imgs = -1):

	encoding_model = load_encoding_model()

	#for img in train_imgs:
	encoded_image = get_encoding(encoding_model, image_name)
	
	return encoded_image

	#with open( "encoded_images_resnet.p", "wb" ) as pickle_f:
	#	pickle.dump( encoded_images, pickle_f )  
'''
if __name__ == '__main__':
	image_name = "basketBall.jpg"
	prepare_dataset(image_name="basketBall.jpg")
'''
