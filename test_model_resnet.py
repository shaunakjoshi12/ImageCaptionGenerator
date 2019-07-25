import cPickle as pickle
import caption_generator_resnet
import numpy as np
from keras.preprocessing import sequence
import nltk
import datetime
from get_encoding_resnet import prepare_image_encoding

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

def test_model(weight, img_name, beam_size = 3):
	#encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)

	#image = encoded_images[img_name]
	image = prepare_image_encoding(image_name=img_name)
	
	captions = generate_captions(model, image, beam_size)
	return process_caption(get_best_caption(captions))
	#return [process_caption(caption[0]) for caption in get_all_captions(captions)] 

def bleu_score(hypotheses, references, weights):
	return nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights)

def test_model_on_images(weight, img_dir, beam_size = 3):
	imgs = []
	captions = {}
	with open(img_dir, 'rb') as f_images:
		imgs = f_images.read().strip().split('\n')
	#encoded_images = pickle.load( open( "encoded_images_resnet.p", "rb" ) )
	encoded_images = cg.encoded_images
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)

	f_pred_caption = open('predicted_captions_256_lstm_512_k0.8_b512_00001_beam3_e606.txt', 'wb')

	for count, img_name in enumerate(imgs):
		#print "Predicting for image: "+str(count)
		image = encoded_images[img_name]
		image_captions = generate_captions(model, image, beam_size)
		best_caption = process_caption(get_best_caption(image_captions))
		captions[img_name] = best_caption
		print "Image: ", str(count), img_name, ": ", str(best_caption)
		f_pred_caption.write(img_name+"\t"+str(best_caption)+"\n")
		f_pred_caption.flush()
	f_pred_caption.close()

	f_captions = open('Flickr8k_text/Flickr8k.token.txt', 'rb')
	captions_text = f_captions.read().strip().split('\n')
	image_captions_pair = {}
	for row in captions_text:
		row = row.split("\t")
		row[0] = row[0][:len(row[0])-2]
		try:
			image_captions_pair[row[0]].append(row[1])
		except:
			image_captions_pair[row[0]] = [row[1]]
	f_captions.close()
	
	hypotheses=[]
	references = []
	for img_name in imgs:
		hypothesis = captions[img_name]
		reference = image_captions_pair[img_name]
		hypotheses.append(hypothesis)
		references.append(reference)
		
	with open( "predicted_captions.p", "wb" ) as pickle_f:
		pickle.dump( captions, pickle_f )
		
	with open( "actual_captions.p", "wb" ) as pickle_f:
		pickle.dump( image_captions_pair, pickle_f )

	# bleuScore1 = bleu_score(hypotheses, references, weights=(1, 0, 0, 0))
	# bleuScore2 = bleu_score(hypotheses, references, weights=(0.5, 0.5, 0, 0))
	# bleuScore3 = bleu_score(hypotheses, references, weights=(0.34, 0.33, 0.33, 0))
	# bleuScore4 = bleu_score(hypotheses, references, weights=(0.25, 0.25, 0.25, 0.25))
	bleuScore1 = bleu_score(references,hypotheses, weights=(1, 0, 0, 0))
	bleuScore2 = bleu_score(references,hypotheses, weights=(0.5, 0.5, 0, 0))
	bleuScore3 = bleu_score(references,hypotheses, weights=(0.34, 0.33, 0.33, 0))
	bleuScore4 = bleu_score(references,hypotheses, weights=(0.25, 0.25, 0.25, 0.25))	
	return bleuScore1, bleuScore2, bleuScore3, bleuScore4

if __name__ == '__main__':
	weight = 'Keep_70_dim_512/weights-resnet-256-lstm-512-keep-0.8-B512_L00001_600_06.hdf5'
	
#	test_image = 'test1.jpg'
	test_img_dir = 'Flickr8k_text/Flickr_8k.testImages.txt'
	
	startTime = datetime.datetime.now()
#	predictedCaptions =(weight, test_image)
	bleuScore1, bleuScore2, bleuScore3, bleuScore4 = test_model_on_images(weight, test_img_dir, beam_size=3)
	endTime = datetime.datetime.now()
	print "\nTest Weights: ", weight
	print "Start Time: ", startTime
	print "End Time: ", endTime
	print "Time Taken: ", endTime-startTime
#	print "Predicted Captions: ", predictedCaptions

	print "\nBLEU-1 Score of the Model: ", bleuScore1
	print "BLEU-2 Score of the Model: ", bleuScore2
	print "BLEU-3 Score of the Model: ", bleuScore3
	print "BLEU-4 Score of the Model: ", bleuScore4
