
#93 Deep learning using linear support vector machines使用线性支持向量机训练fer2013
#Y . Tang, “Deep learning using linear support vector machines,”arXiv
# preprint arXiv:1306.0239, 2013.
import tensorflow as tf
import numpy as np
import pandas as pd
import os
# ----------------------模型----------------------------------------
class CNN(tf.keras.Model):
    def __init__(self, num_class, keep_prob):
        super(CNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, use_bias=True, padding='same')
        self.conv1_act = tf.keras.layers.Activation('relu')
        self.conv1_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')

        self.conv2 = tf.keras.layers.Conv2D(64, 5, strides=2, use_bias=True, padding='same')
        self.conv2_act = tf.keras.layers.Activation('relu')
        self.conv2_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')

        self.flat = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(1024, activation='relu', use_bias=True)
        self.dense1_act = tf.keras.layers.Activation('relu')

        self.drop = tf.keras.layers.Dropout(rate=keep_prob)

        self.dense2 = tf.keras.layers.Dense(num_class, use_bias=True)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv1_act(x)
        x = self.conv1_pool(x)

        x = self.conv2(x)
        x = self.conv2_act(x)
        x = self.conv2_pool(x)

        x = self.flat(x)

        x = self.dense1(x)
        x = self.dense1_act(x)
        x = self.drop(x)
        x = self.dense2(x)
        return x


# ---------------------------参数---------------------
model_path = r'./checkpoint/emotion_analysis.ckpt'  # path where to save the trained model
penalty_parameter = 0.02  # the SVM C penalty parameter
log_path = r'/logs/'  # path where to save the TensorBoard logs
num_classes = 7
dropout_rate = 0.5  # dropout
batch_size = 128
epoch = 100
lr = 3e-4
weight_decay = 1e-4
data_path = r'../Datasets/archive/fer2013.csv'

AUTOTUNE = tf.data.experimental.AUTOTUNE
np.set_printoptions(precision=3, suppress=True)

# 读取csv文件
df = pd.read_csv(filepath_or_buffer=data_path, usecols=["emotion", "pixels"], dtype={"pixels": str})
fer_pixels = df.copy()

# 分成特征和标签
fer_label = fer_pixels.pop('emotion')
fer_pixels = np.asarray(fer_pixels)

# 将特征转换成模型需要的类型
fer_train = []
for i in range(len(fer_label)):
    pixels_new = np.asarray([float(p) for p in fer_pixels[i][0].split()]).reshape([48,48,1])
    fer_train.append(pixels_new)
fer_train = np.asarray(fer_train)
fer_label = np.asarray(fer_label)

# 转换为tf.Dateset类型
dataset = tf.data.Dataset.from_tensor_slices((fer_train, fer_label))

# 数据集验证集测试集的拆分
train_dataset = dataset.take(1000)  # 训练集
test_dataset = dataset.skip(32297)

# 打乱
train_dataset = (train_dataset.cache().shuffle(5 * batch_size).batch(batch_size).prefetch(AUTOTUNE))

# 训练
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = CNN(num_class=num_classes, keep_prob=dropout_rate)
    model.compile(loss=tf.keras.losses.Hinge(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    # 断点续训
    if os.path.exists(model_path+'/saved_model.pb'):
        print('-----------------加载模型---------------------')
        model = tf.keras.models.load_model(model_path)#加载model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                     save_weights_only=False,
                                                     monitor='val_accuracy',
                                                     model='max',
                                                     save_best_only=True)
    # 训练
    history = model.fit(x=train_dataset, epochs=epoch,callbacks=[cp_callback])
    model.summary()


# if __name__ == "__main__":
#     (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
#     x_train = x_train.reshape((60000,28,28,1))
#     x_test = x_test.reshape((10000,28,28,1))
#
#     x_train,x_test = x_train/255.0,x_test/255.0
#
#     model = CNN(num_class=7,keep_prob=0.5)
#
#     model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#     model.fit(x=x_train, y=y_train, epochs=5)


