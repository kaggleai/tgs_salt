from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
#import Augmentor
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Model, load_model 
from keras.layers import Activation, Input, Concatenate, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from datetime import datetime




#################### Creating directories ####################
def create_dir(list_of_directories, parent_directory = ".."):
    """
    Creates directories. Use it to generate directories to store models, submission files, etc.
    
    input
    list_of_directories: a list of strings (paths to directories to be created)
    parent_directory: a string. Parent directory of each directory in list_of_directories
    
    return
    None
    """
    for directory in list_of_directories:
        path = os.path.join(parent_directory, directory)
        if not directory in os.listdir(parent_directory):
            print("Creating directory " + path + "...")
            os.mkdir(path)
            print("Done!")
        else:
            print("Directory " + path + " already exists.")
    return None
        




#################### Processing data ####################

def load_data(path_to_images, path_to_masks = None, normalize = False):
    """
    Loads training and testing image data.
    
    input
    path_to_images: a string. Path to images
    path_to_masks: a string. Path to masks
    normalize: a boolean. Whether to normalize images or not

    return
    X_train: a numpy array. Training set
    y_train: a numpy array. Training targets
    """
        
    input_dimensions = (101,101,1); output_dimensions = (101,101,1)
    ids = os.listdir(path_to_images)
    
   
    X_train = np.zeros((len(ids), *output_dimensions), dtype = np.uint8)
    if path_to_masks is not None:
        y_train = np.zeros((len(ids), *output_dimensions), dtype = np.bool)
            
    for ix, id_ in tqdm(enumerate(ids), total = len(ids)):
        image = load_img(os.path.join(path_to_images, id_))
        x = img_to_array(image)[:,:,0]
        x = resize(x, output_dimensions, mode = "constant", preserve_range = True)
        X_train[ix] = x
        if path_to_masks is not None:
            mask = load_img(os.path.join(path_to_masks, id_))
            y = img_to_array(mask)[:,:,0]
            y = resize(y, output_dimensions, mode = "constant", preserve_range = True)
            y_train[ix] = y
    if normalize:
        X_train = X_train/255
    if path_to_masks is not None:    
        return X_train, y_train
    else:
        return X_train

def minimal_aug(image_data, mask_data, shuffle = True, random_state = 131):
    """
    input
    image_data: a numpy array. Images to be augmented
    mask_data: a numpy array. Masks to be augmented
    shuffle: boolean. Whether to shuffle the data or not
    augmentations: a list of strings. One of the eight possible augmentation operations
    random_state: seed for random number generator
 
    return
    X: a numpy array. Augmented image data
    y: a numpy array. Augmented mask data
    """

    
    #dict_image = {"horizontal_flip": np.fliplr(image_data), "vertical_flip": np.flipud(), "a":3}
    np.random.seed(random_state)
    augmented_images = np.zeros((len(image_data)*8, *image_data.shape[1:]), dtype = np.uint8)
    augmented_masks = np.zeros((len(mask_data)*8, *mask_data.shape[1:]), dtype = np.bool)
    #print("shape augmented_images: ", augmented_images.shape)
    #print("shape augmented_masks: ", augmented_masks.shape)
    output_dimensions = image_data.shape[1:]
    #print("output_dimensions: ", output_dimensions)
    for ix in tqdm(range(len(image_data)), desc = "Augmenting images"):
        x = image_data[ix][:,:,0]
        y = mask_data[ix][:,:,0]
        #print("shape of x: ", x.shape)
        #print("shape of y: ", y.shape)
        ###identity
        augmented_images[8*ix] = resize(x, output_dimensions, mode = "constant", preserve_range = True)
        augmented_masks[8*ix] = resize(y, output_dimensions, mode = "constant", preserve_range = True)
        ###horizontal flip
        augmented_images[8*ix+1] = resize(np.fliplr(x), output_dimensions, mode = "constant", preserve_range = True)
        augmented_masks[8*ix+1] = resize(np.fliplr(y), output_dimensions, mode = "constant", preserve_range = True)
        ###vertical flip
        augmented_images[8*ix+2] = resize(np.flipud(x), output_dimensions, mode = "constant", preserve_range = True)
        augmented_masks[8*ix+2] = resize(np.flipud(y), output_dimensions, mode = "constant", preserve_range = True)
        ###transposition
        augmented_images[8*ix+3] = resize(x.T, output_dimensions, mode = "constant", preserve_range = True)
        augmented_masks[8*ix+3] = resize(y.T, output_dimensions, mode = "constant", preserve_range = True)
        ###anti_transposition
        augmented_images[8*ix+4] = resize(x[::-1,::-1].T, output_dimensions, mode = "constant", preserve_range = True)
        augmented_masks[8*ix+4] = resize(y[::-1,::-1].T, output_dimensions, mode = "constant", preserve_range = True)
        ###rotation90
        augmented_images[8*ix+5] = resize(np.rot90(x,k=1), output_dimensions, mode = "constant", preserve_range = True)
        augmented_masks[8*ix+5] = resize(np.rot90(y,k=1), output_dimensions, mode = "constant", preserve_range = True)
        ###rotation180
        augmented_images[8*ix+6] = resize(np.rot90(x,k=2), output_dimensions, mode = "constant", preserve_range = True)
        augmented_masks[8*ix+6] = resize(np.rot90(y,k=2), output_dimensions, mode = "constant", preserve_range = True)
        ###rotation270
        augmented_images[8*ix+7] = resize(np.rot90(x,k=3), output_dimensions, mode = "constant", preserve_range = True)
        augmented_masks[8*ix+7] = resize(np.rot90(y,k=3), output_dimensions, mode = "constant", preserve_range = True)
    if shuffle:
        shuffled_indices = np.random.permutation(len(augmented_images))
        augmented_images = augmented_images[shuffled_indices]
        augmented_masks = augmented_masks[shuffled_indices]
    return augmented_images, augmented_masks
        



def make_validation(training_data, val_split = 0.1, random_state = None):
    """
    Creates training and validation sets.

    input
    training_data: a tuple (X_train, y_train)
    val_split: a float between 0 and 1. Fraction of data used for validation    
    
    return
    X_train: training data
    y_train: training_targets
    X_val: validation data
    y_val: validation targets
    """
    if random_state is not None:
        np.random.seed(random_state)
    X_train, y_train = training_data
    shuffle = np.random.permutation(len(X_train))
    train_indices = shuffle[:int(len(X_train)*(1-val_split))]
    val_indices = shuffle[int(len(X_train)*(1-val_split)):]
    X_val, y_val = X_train[val_indices], y_train[val_indices] 
    X_train, y_train = X_train[train_indices], y_train[train_indices]
    return X_train, y_train, X_val, y_val
        


#np.random.seed(32)
#train_split = 0.9
#shuffle = np.random.permutation(len(X_train))
#train_indices = shuffle[:int(len(X_train)*train_split)]
#val_indices = shuffle[int(len(X_train)*train_split):]


#X_val, y_val = X_train[val_indices], y_train[val_indices]
#X_train, y_train = X_train[train_indices], y_train[train_indices]

#print("Shape of X_train: ", X_train.shape, "\t Shape of y_train: ", y_train.shape)
#print("Shape of X_val: ", X_val.shape, "\t Shape of y_val: ", y_val.shape)

def upsample(data_array):
    """
    Removes channel dimension from array of images.
    
    input
    data_array: an array of images, each being of shape (height,width,channel)
    
    return
    upsampled: an array without channel dimension
    """
    return np.squeeze(data_array, axis=2)

        


#################### Training ####################


def mean_iou(y_true, y_pred):
    """
    Computes mean IoU score.
    
    input
    y_true: an array of true target labels
    y_pred: an array of predicted target labels

    return
    mean_iou_score = computed mean IoU score
    
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    mean_iou_score = K.mean(K.stack(prec), axis=0)
    #return K.mean(K.stack(prec), axis=0)
    return mean_iou_score


def make_batches(training_data, sequence = None):
    """
    Make training batches with augmentation.

    input
    sequence: an augmentation object from imgaug containing a sequence of augmentation operations
    training_data: a tuple (X_train, y_train)

    return
    X_batch: a numpy array. Training batch
    y_batch: a numpy array. Target batch
    """
    X_batch, y_batch = training_data
    #y = y_batch.astype(np.int)*255
    y_batch = y_batch.astype(np.uint8)*255
    #print("max y: ", np.max(y))
    if sequence is not None:
        aug = sequence._to_deterministic()
        #X_batch, y_batch = aug.augment_images(X_batch), aug.augment_images(y_batch)
        X_batch, y_batch = aug.augment_images(X_batch), aug.augment_images(y_batch)
    #y_batch = np.rint(y_batch/255).astype(np.int)
    #y_batch = y_batch.astype(np.bool)
    #y_batch = np.copy( np.rint(y_batch/255).astype(np.int).astype(np.bool) )
    #print("max y after augmentation: ", np.max(y))
    #print("")
    #y_batch =  np.copy( np.rint(y/255).astype(np.int)).astype(np.bool)
    #print("max y_batch after rounding: ", np.max(y_batch))
    #y_batch = np.copy(y_batch.astype(np.bool))
    y_batch =  ( np.rint(y_batch/255).astype(np.int)).astype(np.bool)
    return X_batch, y_batch
   

#################### Neural Networks ####################
class NeuralNet(object):
    """
    A Python class that stores the neural networks for this work.
    """
    def __init__(self, input_dim = (101,101,1), output_dim = (101,101,1),parent_dir = "../models", model_name = "jesper",\
                load = False):
        """
        NeuralNet class constructor

        input
        input_dim: tuple-like containing dimensions of input arrays as (height, width, channel)
        output_dim: tuple-like containing dimensions of output arrays as (height, width, channel)
        """
        #now = datetime.now()
        #timestamp = "_".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute)])
        #self.timestamp = "_".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute)])
        self.parent_dir = parent_dir
        self.model_name = model_name
        #print("modelname: ", modelname)
        #print("self.modelname: ", self.modelname)
        if load:
            filename = self.model_name + ".h5"
            self.model = load_model(os.path.join(self.parent_dir, filename), \
                                      custom_objects={"mean_iou": mean_iou})             
        else:
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.inputs = Input(self.input_dim)
            self.s = Lambda(lambda x: x/255)(self.inputs)
            self.outputs = None
            self.model = None
            

    
    def summary(self):
        """
        Prints model summary.
        """
        self.model.summary()
        

    def compile(self,dict_compile):
        """
        Compiles a model with dict_args parameters
        input
        dict_args: a dictionary of compilation arguments.
        """
        self.model.compile(**dict_compile)
    
    
    def fit(self, dict_fit, load_best_model = True):
        """
        Fits model with dict_args parameters
        
        input
        dict_fit: a dictionary of fitting parameters
        """
        filename = self.model_name + ".h5"
        path_to_save = os.path.join(self.parent_dir, filename)
        checkpointer = ModelCheckpoint(path_to_save, verbose = 1, save_best_only = True)
        dict_fit["callbacks"][1] = checkpointer
        print("Fitting model...")
        self.model.fit(**dict_fit)
        if load_best_model:
            print("Loading best model...")   
            self.model = load_model(path_to_save, custom_objects={"mean_iou": mean_iou})
        print("Done.")
        
    
    def predict(self, dict_pred, threshold = 0.5):
        """
        Predicts with dict_pred arguments
        """
        predicted = self.model.predict(**dict_pred)
        predicted_mask = predicted > threshold
        return predicted_mask
    
           

class Jesper(NeuralNet):
    """
    Implements Jesper's unet.
    """

    def __init__(self,input_dim = (101,101,1) , output_dim = (101,101,1),parent_dir = "../models", model_name = "jesper",\
                load = False):
        """
        Generates Jesper's model.
        """
        NeuralNet.__init__(self,input_dim = input_dim, output_dim = output_dim, parent_dir = parent_dir, model_name = model_name, \
                          load = load)
        ###neural network
        #if load:
        #    filename = self.model_name + ".h5"
        #    self.model = load_model(os.path.join(self.parent_dir, filename), \
        #                              custom_objects={"mean_iou": mean_iou}) 
        #else:
        if not load:
            conv1 = Conv2D(filters = 8, kernel_size = (2,2), strides = (1,1), activation = "relu", padding = "same")(self.s)
            batch1 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv1)
            conv1 = Conv2D(filters = 8, kernel_size = (2,2), strides = (1,1), activation = "relu", padding = "same")(batch1)
            batch1 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv1) 
            pool1 = MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "valid")(batch1)

            conv2 = Conv2D(filters = 16, kernel_size = (2,2), strides = (1,1), activation = "relu", padding = "same")(pool1)
            batch2 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv2)
            conv2 = Conv2D(filters = 16, kernel_size = (2,2), strides = (1,1), activation = "relu", padding = "same")(batch2)
            batch2 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv2)
            pool2 = MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "same")(batch2)

            conv3 = Conv2D(filters = 32, kernel_size = (2,2), strides = (1,1), activation = "relu", padding = "same")(pool2)
            batch3 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv3)
            conv3 = Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "same")(batch3)
            batch3 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv3)
            pool3 = MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "valid")(batch3)

            conv4 = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "same")(pool3)
            batch4 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv4)
            conv4 = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "same")(batch4)
            batch4 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv4)
            pool4 = MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "same")(batch4)

            conv5 = Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding='same') (pool4)
            batch5 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv5)
            conv5 = Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding='same') (batch5)
            batch5 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv5)

            trans6 = Conv2DTranspose(filters = 64, kernel_size = (2,2), strides = (2,2), activation = "relu", padding = "same")(batch5)
            batch6 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(trans6)
            u6 = Concatenate(axis = 3)([batch6, batch4])
            conv6 = Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='same') (u6)
            batch6 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv6)
            conv6 = Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='same') (batch6)
            batch6 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv6)

            trans7 = Conv2DTranspose(filters = 32, kernel_size = (3,3), strides = (2,2), \
                                     activation = "relu", padding = "valid")(batch6)
            batch7 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(trans7)
            u7 = Concatenate(axis = 3)([batch7, batch3])
            conv7 = Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding='same') (u7)
            batch7 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv7)
            conv7 = Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding='same') (batch7)
            batch7 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv7)

            trans8 = Conv2DTranspose(filters = 16, kernel_size = (2,2), strides = (2,2), activation = "relu", padding = "same")(batch7)
            batch8 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(trans8)
            u8 = Concatenate(axis = 3)([batch8, batch2])
            conv8 = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same') (u8)
            batch8 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv8)   
            conv8 = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same') (batch8)
            batch8 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv8)

            trans9 = Conv2DTranspose(filters = 8, kernel_size = (3,3), strides = (2,2), activation = "relu", padding = "valid")(batch8)
            batch9 = BatchNormalization(axis = 3, momentum=0.99, epsilon=0.001, center=True, scale=True)(trans9)

            self.outputs = Conv2D(filters = 1, kernel_size = (1,1), activation = "sigmoid")(batch9)
            self.model = Model(inputs = [self.inputs], outputs = [self.outputs])



    


        
            
#################### Ensemble ####################
class Ensemble(object):
    """
    Implements an ensemble of neural networks.
    """
    networks = {"jesper": Jesper}
    
    def __init__(self, n_models = 2, input_dim = (101,101,1) , output_dim = (101,101,1) , network = "jesper", \
                 parent_dir = "../models", model_name = "jesper_ensemble", load = False):
        """
        Class constructor.
        
        input
        network: a string corresponding to an implemented class of neural networks
        n_models: an int. Number of models in ensemble
        saved_models: a list of strings (filenames). These filenames are models to be loaded
        input_dim: a tuple. Dimension of images
        output_dim: a tuple. Dimension of masks
        """
        #now = datetime.now()
        #timestamp = "_".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute)])
        #self.timestamp = "_".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute)])      
        self.parent_dir = parent_dir
        self.model_name = model_name
        self.model_names = [self.model_name + "_" + str(i) for i in range(n_models)]
        #self.saved_models = saved_models
        if load:
            
            filenames = [name + ".h5" for name in self.model_names] 
            #self.model = load_model(os.path.join(self.parent_dir, filename), \
            #                          custom_objects={"mean_iou": mean_iou})  
            
            #all_files = os.listdir(self.parent_dir)
            #model_names = [name for name in all_files if self.model_name in name[:len(self.model_name)]] 
            #model_names = [name for name in all_files if self.model_name in name]  
            print("Loading models...\n")
            #self.model = [load_model(os.path.join(self.parent_dir, model_name), \
            #                          custom_objects={"mean_iou": mean_iou}) for model_name in model_names]
            self.model = []
            for ix, name in tqdm(enumerate(filenames), total = n_models):
                model = load_model(os.path.join(self.parent_dir, name),  custom_objects={"mean_iou": mean_iou})
                self.model.append(model)
            print("Done.\n")
        else:
            self.model = [Ensemble.networks[network](input_dim, output_dim, parent_dir = self.parent_dir, model_name = self.model_names[i])\
                         for i in range(n_models)]
            
            
        
    def compile(self, dict_compile):
        """
        Compiles all models in an ensemble.
        
        input
        dict_compile: a dictionary of compilation arguments
        """
        for model in tqdm(self.model,desc="Compiling models"):
            model.compile(dict_compile)
            
    def fit(self, dict_fit, load_best_model = True, sample = None, random_state = 538):
        """
        Fits all models in an ensemble
        
        input
        dict_fit: a dictionary of fitting arguments
        sample: a float. Fraction of data to use to train each model
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        #now = datetime.now()
        #timestamp = "_".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute)])
        #parent_directory = "../models"
        if sample is not None:   #sample is a float (fraction of data)
            X = dict_fit["x"]
            y = dict_fit["y"]
            len_data = len(X)
            len_sample = int(sample*len_data)  #sample is a float (fraction of data)
        
        #print("Fitting models...")
        for ix, model in tqdm(enumerate(self.model), desc = "Fitting models", total = len(self.model)):
            #model_name = model_name + "_" + str(ix) 
            path_to_save = os.path.join(self.parent_dir, self.model_name[ix])
            checkpointer = ModelCheckpoint(path_to_save, verbose = 1, save_best_only = True)
            #checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True)
            #dict_fit["callbacks"][1] = checkpointer
            if sample is not None:
                shuffle = np.random.permutation(len_data)[:len_sample] 
                X_train = X[shuffle]
                y_train = y[shuffle]
                if "validation_data" not in dict_fit:
                    assert "validation_split" in dict_fit, "You have to provide either validation_data or validation_split in dict_fit"
                    #val_split = dict_fit["validation_split"]
                    #X_val = X_train[int(len_sample*(1-val_split)):]
                    #y_val = y_train[int(len_sample*(1-val_split)):]
                    #X_train = X_train[:int(len_sample*(1-val_split))]
                    #y_train = y_train[:int(len_sample*(1-val_split))]
                    #dict_fit["validation_data"] = (X_val, y_val)
                dict_fit["x"] = X_train
                dict_fit["y"] = y_train
            elif "validation_data" not in dict_fit:
                assert "validation_split" in dict_fit, "You have to provide either validation_data or validation_split in dict_fit"
            model.fit(dict_fit, load_best_model = False)
            #self.modelname.append(model.modelname)
        #self.models = []
        #for modelname in self.saved_models:
        #    model = load_model(modelname, custom_objects={"mean_iou": mean_iou})
        #    self.models.append(model)
        if load_best_model:
            print("Loading models...")
            self.model = [load_model(os.path.join(self.parent_dir, model_name + ".h5"),\
                                  custom_objects={"mean_iou": mean_iou}) for model_name in self.model_names]
        print("Done.")
    
    
    def predict(self, dict_pred, threshold = 0.5):
        predictions = []
        for model in tqdm(self.model, desc = "Predicting"):
            predicted = model.predict(**dict_pred)
            predictions.append(predicted)
        predicted_mask = (sum(predictions)/len(predictions)) > threshold
        return predicted_mask 

    #def load_predict(self, dict_predict, threshold = 0.5, model_name = "jesper_ensemble"):
    #    all_files = os.listdir(self.parent_dir)
    #    model_names = [name for name in all_files if self.model_name in name] 

        
    #    if load:
    #        all_files = os.listdir(self.parent_dir)
    #        #model_names = [name for name in all_files if self.model_name in name[:len(self.model_name)]] 
    #        model_names = [name for name in all_files if self.model_name in name]  
    #        print("Loading models...\n")
    #        self.model = [load_model(os.path.join(self.parent_dir, model_name), \
    #                                  custom_objects={"mean_iou": mean_iou}) for model_name in model_names]
    #        print("Done.\n")
        
        
        
            
        
        
        
        
 



        
 
    
#################### Submission ####################

def format_mask(mask_array, order = "F", format = True):
    """
    Generates a submision csv for mask_data
  
    input
    mask_array: a single binary mask array
    """
    bytes = mask_array.reshape(mask_array.shape[0] * mask_array.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def make_prediction_file(path_to_images, masks_array, submissions_dir = "../submissions", submission_name="submission.csv"):
    """
    Generates dictionary of predictions in proper format
    
    input
    path_to_images: a string. Path to images
    masks_array: array of binary masks
    path_to_save: a string. Path of saved file
    """
    #now = datetime.now()
    #timestamp = "_".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute)])
    #submission_name = "submission_" + timestamp + "--" + str(ix) + ".csv"
    path_to_save = os.path.join(submissions_dir, submission_name)
    ids = os.listdir(path_to_images)
    pred_dict = {id_[:-4]: format_mask(np.squeeze(masks_array[ix], axis=2)) for ix, id_ in tqdm(enumerate(ids), total = len(ids))}
    sub = pd.DataFrame.from_dict(pred_dict, orient = "index")
    sub.index.names = ["id"]
    sub.columns = ["rle_mask"]
    sub.to_csv(path_to_save)
    return sub
    
