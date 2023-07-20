import keras

class ConvModel(keras.Model):
    
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1))
        self.pool1 = keras.layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same')

        self.conv2 = keras.layers.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool2 = keras.layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same')

        self.conv3 = keras.layers.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool3 = keras.layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same')
        self.drop1 = keras.layers.Dropout(0.2)

        self.conv4 = keras.layers.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool4 = keras.layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same')
        
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(units=32, activation='relu')
        self.drop2 = keras.layers.Dropout(0.3)

        self.dense2 = keras.layers.Dense(units=8, activation='softmax')

    def call(self, inputs):

        x = self.conv1(inputs)


model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()