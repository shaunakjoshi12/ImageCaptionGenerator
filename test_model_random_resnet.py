import cPickle as pickle
import caption_generator_resnet
import numpy as np
from keras.preprocessing import sequence
import nltk
import datetime
from get_encoding_resnet import prepare_image_encoding
from keras.models import load_model
import socket

cg = caption_generator_resnet.CaptionGenerator()

def process_caption(caption):
	caption_split = caption.split()
	processed_caption = caption_split[1:]
	try:
		end_index = processed_caption.index('<end>')
		processed_caption = processed_caption[:end_index]
	except:
		pass
	return " ".join([word for word in processed_caption])

def get_best_caption(captions):
    captions.sort(key = lambda l:l[1])
    best_caption = captions[-1][0]
    return " ".join([cg.index_word[index] for index in best_caption])

def get_all_captions(captions):
    final_captions = []
    captions.sort(key = lambda l:l[1])
    for caption in captions:
        text_caption = " ".join([cg.index_word[index] for index in caption[0]])
        final_captions.append([text_caption, caption[1]])
    return final_captions

def generate_captions(model, image, beam_size):
	start = [cg.word_index['<start>']]
	captions = [[start,0.0]]
	while(len(captions[0][0]) < cg.max_cap_len):
		temp_captions = []
		for caption in captions:
			partial_caption = sequence.pad_sequences([caption[0]], maxlen=cg.max_cap_len, padding='post')
			next_words_pred = model.predict([np.asarray([image]), np.asarray(partial_caption)])[0]
			next_words = np.argsort(next_words_pred)[-beam_size:]
			for word in next_words:
				new_partial_caption, new_partial_caption_prob = caption[0][:], caption[1]
				new_partial_caption.append(word)
				new_partial_caption_prob+=next_words_pred[word]
				temp_captions.append([new_partial_caption,new_partial_caption_prob])
		captions = temp_captions
		captions.sort(key = lambda l:l[1])
		captions = captions[-beam_size:]
	return captions

def test_model(img_name, beam_size = 3):
	
	image = prepare_image_encoding(image_name=img_name)
	
	captions = generate_captions(model, image, beam_size)
	return process_caption(get_best_caption(captions))

if __name__ == '__main__':
	weight = 'resnet1024_532.hdf5'
	
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)
	
	socket = socket.socket()
	socket.bind(("localhost", 8888))
	socket.listen(5)
	
	while True:
		c, addr = socket.accept()
		print "Connection Establish with client: ", addr
		
		test_image = c.recv(1024)
		# "/var/www/html/GUI/uploads/"+
		startTime = datetime.datetime.now()
		predictedCaptions = test_model(test_image)
		endTime = datetime.datetime.now()
		
		c.send(predictedCaptions + "\t" + str(endTime-startTime))
		
		print "Start Time: ", startTime
		print "End Time: ", endTime
		print "Predicted Captions: ", predictedCaptions
		
		c.close()
