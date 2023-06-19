import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from model import Xception
import argparse
from keras_radam import RAdam
from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from Video_Generator import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xception model")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[50, 120, 160, 3], help="Input shape")
    #parser.add_argument("--num_classes", type=int, default=1, help="Number of classes")
    #parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    #parser.add_argument("--l1", type=float, default=0.001, help="L1 regularization")
    #parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization")

    args = parser.parse_args()

    #model = Xception(input_shape=args.input_shape, num_classes=args.num_classes, dropout_rate=args.dropout_rate, l1=args.l1, l2=args.l2)
    model = Xception(input_shape=args.input_shape)

    model.summary()

    opt = RAdam(lr=0.0001, decay=0.5)

    rmse = RootMeanSquaredError()
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', rmse])



    # every epoch check validation accuracy scores and save the highest
    checkpoint_2 = ModelCheckpoint(
        'weights-{epoch:02d}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_root_mean_squared_error',
        verbose=1, save_best_only=True)
    # every 10 epochs save weights
    checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_root_mean_squared_error:.4f}.h5',
                                 monitor='val_mean_absolute_error',
                                 verbose=10, save_best_only=True)
    history_checkpoint = CSVLogger("history.csv", append=False)

    # use tensorboard can watch the change in time
    tensorboard_ = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=5, verbose=1, mode='auto')

    """
    if (CONTINUE_TRAINING == True):
        history = pd.read_csv('history.csv')
        INITIAL_EPOCH = history.shape[0]
        model.load_weights('weights_%02d.h5' % INITIAL_EPOCH)
        checkpoint_2.best = np.min(history['val_root_mean_squared_error'])
    else:
        INITIAL_EPOCH = 0
    """

    datagen = ImageDataGenerator()

    batch_size_train = 1
    batch_size_test = 1

    train_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/BP4D-ROI',
                                             label_dir='/home/ouzar1/Documents/pythonProject/Physiology',
                                             target_size=(120, 160), class_mode='label', batch_size=64,
                                             frames_per_step=50, shuffle=False)

    test_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/MMSE/ROI',
                                            label_dir='/home/ouzar1/Documents/MMSE/HR',
                                            target_size=(120, 160), class_mode='label', batch_size=64,
                                            frames_per_step=50, shuffle=False)

    history = model.fit(train_data, epochs=100,
                        steps_per_epoch=len(train_data.filenames) // 3200,
                        validation_data=test_data, validation_steps=len(test_data.filenames) // 3200,
                        callbacks=[history_checkpoint, checkpoint_2])



    values = history.history
    validation_loss = values['val_loss']
    validation_mae = values['val_mae']
    training_mae = values['mae']
    validation_rmse = values['val_root_mean_squared_error']
    training_rmse = values['root_mean_squared_error']
    training_loss = values['loss']

    epochs = range(100)

    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.title('Epochs vs Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, training_mae, label='Training MAE')
    plt.plot(epochs, validation_mae, label='Validation MAE')
    plt.title('Epochs vs MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

    plt.plot(epochs, training_rmse, label='Training RMSE')
    plt.plot(epochs, validation_rmse, label='Validation RMSE')
    plt.title('Epochs vs RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()
    plt.show()


