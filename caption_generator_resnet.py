#from vgg16 import VGG16

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten, Dropout
from keras.preprocessing import image, sequence
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop
import cPickle as pickle

EMBEDDING_DIM = 256


class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.encoded_images = pickle.load( open( "encoded_images_resnet.p", "rb" ) )
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = iter.next()
            caps.append(x[1][1])

        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())
        print "Total samples : "+str(self.total_samples)
        
        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word]=i
            self.index_word[i]=word

        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print "Vocabulary size: "+str(self.vocab_size)
        print "Maximum caption length: "+str(self.max_cap_len)
        print "Variables initialization done!"


    def data_generator(self, batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        print "Generating data..."
        gen_count = 0
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = iter.next()
            caps.append(x[1][1])
            imgs.append(x[1][0])


        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next = np.zeros(self.vocab_size)
                    next[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next)
                    images.append(current_image)
        
                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        #print "yielding count: "+str(gen_count)
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
        
    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)

    def create_model(self, ret_model = False): #image model is removed because images encodings are directly taken file
        #base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
        #base_model.trainable=False
        
        image_model = Sequential()
        #image_model.add(base_model)
        #image_model.add(Flatten())
        image_model.add(Dense(EMBEDDING_DIM, input_dim = 2048, activation='relu'))
        #image_model.add(RepeatVector(self.max_cap_len))
        image_model.add(RepeatVector(1))
        print "--------------------------------------------Image model summary---------------------------------------------"
	print image_model.summary()

        lang_model = Sequential()
        lang_model.add(Embedding(self.vocab_size, 512, input_length=self.max_cap_len))
        lang_model.add(LSTM(512,return_sequences=True))
        lang_model.add(Dropout(0.2))
        lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))
        print "--------------------------------------------Language model summary---------------------------------------------"
	print lang_model.summary()

        model = Sequential()
        model.add(Merge([image_model, lang_model], mode='concat', concat_axis=1))
        model.add(LSTM(1000,return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))
        print "--------------------------------------------Whole model summary---------------------------------------------"
	print model.summary()

        print "Model created!"

        if(ret_model==True):
            return model

       # myOptimizer = Adam(lr=0.0001	, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        myOptimizer = RMSprop(lr=0.000005, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=myOptimizer, metrics=['accuracy'])
        return model
    def get_word(self,index):
        return self.index_word[index]
