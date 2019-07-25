import caption_generator_resnet
from keras.callbacks import ModelCheckpoint, Callback
import datetime

class onEpochEndClass(Callback):
    def on_epoch_end(self, epochs, logs={}):
        print datetime.datetime.now(), "\n"
        return

def train_model(weight = None, batch_size=32, epochs = 10):

    cg = caption_generator_resnet.CaptionGenerator()
    model = cg.create_model()

    if weight != None:
        model.load_weights(weight)

    counter = 0
    file_name = 'Keep_70_dim_512/weights-resnet-256-lstm-512-keep-0.8-B1024_L000005_606_{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    onEpochEnd = onEpochEndClass()
    callbacks_list = [checkpoint, onEpochEnd]
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=epochs, verbose=2, callbacks=callbacks_list)
    try:
        model.save('Models/WholeModelResnetLSTM512keep0.6_rmsprop.h5', overwrite=True)
        model.save_weights('Models/WeightsResnetLSTM512keep0.6_rmsprop.h5',overwrite=True)
    except:
        print "Error in saving model."
    print "Training complete...\n"

if __name__ == '__main__':
   # train_model(weight='Keep_70_dim_768_2048/weights-improvement-resnet-lstm-dim-chg-keep-0.6-05.hdf5', epochs=25)
    train_model(weight="Keep_70_dim_512/weights-resnet-256-lstm-512-keep-0.8-B512_L00001_600_06.hdf5", batch_size=512, epochs=100)
