import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO and WARNING messages are not printed
import tensorflow as tf
from keras.layers import Layer,Input,Conv2D,AveragePooling2D,Concatenate,Conv2DTranspose,GlobalAveragePooling2D,Dense,Multiply

# tf.compat.v1.disable_eager_execution()

class EdgeBlock(Layer):
    def __init__(self, filter_size, deconv_kernel, name="edge"):
        super(EdgeBlock, self).__init__()
        self.conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same')
        self.dconv = Conv2DTranspose(filters=filter_size, kernel_size=deconv_kernel, strides=deconv_kernel, padding='same')
        self.conv2 = Conv2D(filters=filter_size, kernel_size=4, strides=1, padding='same', activation='softmax')

    def call(self, inputs):
        x_1 = self.conv1(inputs)
        x_2 = self.dconv(x_1)
        y = self.conv2(x_2)
        return y


class RegionBlock(Layer):
    def __init__(self, filter_size, kernel_size, name="region"):
        super(RegionBlock, self).__init__()
        self.conv1 = Conv2D(filters=filter_size, kernel_size=kernel_size, strides=kernel_size, padding='valid')
        self.conv2 = Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same')
        self.dconv = Conv2DTranspose(filters=filter_size, kernel_size=16, strides=16, padding='same', activation='softmax')

    def call(self, inputs):
        x_1 = self.conv1(inputs)
        x_2 = self.conv2(x_1)
        x_3 = self.conv2(x_2)
        y = self.dconv(x_3)
        return y

class DirectionBlock(Layer):
    def __init__(self, units, name="direction"):
        super(DirectionBlock, self).__init__()
        self.gap = GlobalAveragePooling2D()
        self.dense1 = Dense(units=units, activation='softmax')
        self.dense2 = Dense(units=units, activation='softmax')
        self.dense3 = Dense(units=8, activation='softmax')
        self.multiply = Multiply()

    def call(self, inputs):
        x_1 = self.gap(inputs)
        x_2 = self.dense1(x_1)
        x_3 = self.dense2(x_2)
        x_4 = self.dense3(x_3)
        x_5 = self.multiply([inputs, x_4])
        y = tf.expand_dims(tf.math.reduce_sum(x_5, axis=-1),-1)
        return y

class topology_model(tf.keras.Model):
    def __init__(self):
        super(topology_model,self).__init__(name="topology")
        self.conv1 = Conv2D(filters=3, kernel_size=3, strides=2, padding='same', activation='relu')
        self.avg = AveragePooling2D(pool_size=2, strides=1, padding='same')
        self.conc = Concatenate(axis=-1)
        self.pconv1 = Conv2DTranspose(filters=2048, kernel_size=1, strides=2, padding='valid', activation='relu')
        self.pconv2 = Conv2DTranspose(filters=1024, kernel_size=1, strides=2, padding='valid', activation='relu')
        self.pconv3 = Conv2DTranspose(filters=512, kernel_size=1, strides=2, padding='valid', activation='relu')
        self.pconv4 = Conv2DTranspose(filters=128, kernel_size=1, strides=2, padding='valid', activation='relu')
        self.pconv5 = Conv2DTranspose(filters=32, kernel_size=1, strides=2, padding='valid', activation='relu')
        self.econv = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.e_1 = EdgeBlock(filter_size=8, deconv_kernel=2)
        self.e_2 = EdgeBlock(filter_size=16, deconv_kernel=4)
        self.e_3 = EdgeBlock(filter_size=32, deconv_kernel=8)
        self.e_4 = EdgeBlock(filter_size=64, deconv_kernel=16)
        self.rconv = Conv2D(filters=64, kernel_size=3, padding='same')
        self.r_1 = RegionBlock(filter_size=8, kernel_size=8)
        self.r_2 = RegionBlock(filter_size=16, kernel_size=4)
        self.r_3 = RegionBlock(filter_size=32, kernel_size=2)
        self.r_4 = RegionBlock(filter_size=64, kernel_size=1)
        self.conv2 = Conv2D(filters=8, kernel_size=3, padding='same')
        self.base = tf.keras.applications.resnet50.ResNet50(input_shape=((512,512,3)), include_top=False, weights='imagenet')
        self.base_model = tf.keras.Model(inputs=self.base.inputs,
            outputs=[self.base.get_layer(index=l).output for l in [38,80,142,174]])
        # self.base_layers = [l for l in self.base.layers]
        self.d = DirectionBlock(units=40)
    
    @tf.function
    def call(self, inputs, print_shape_type=False):
        x_1 = self.conv1(inputs); print("x_1: ", x_1.shape, type(x_1)) if print_shape_type else None
        x_2 = self.avg(x_1); print("x_2: ", x_2.shape, type(x_2)) if print_shape_type else None

        # pixel learning
        p = self.base_model(inputs); print("p: ", len(p), type(p)) if print_shape_type else None

        p_5 = self.pconv1(p[3]); print("p_5: ", p_5.shape, type(p_5)) if print_shape_type else None
        p_5 = self.conc([p[2], p_5]); print("p_5: ", p_5.shape, type(p_5)) if print_shape_type else None
        
        p_5 = self.pconv2(p_5); print("p_5: ", p_5.shape, type(p_5)) if print_shape_type else None
        p_5 = self.conc([p[1], p_5]); print("p_5: ", p_5.shape, type(p_5)) if print_shape_type else None
        
        p_5 = self.pconv3(p_5); print("p_5: ", p_5.shape, type(p_5)) if print_shape_type else None
        p_5 = self.conc([p[0], p_5]); print("p_5: ", p_5.shape, type(p_5)) if print_shape_type else None
        
        p_5 = self.pconv4(p_5); print("p_5: ", p_5.shape, type(p_5)) if print_shape_type else None
        p_5 = self.pconv5(p_5); print("p_5: ", p_5.shape, type(p_5)) if print_shape_type else None
        p_5_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(p_5, axis=-1),p_5.shape[-1]),-1); print("p_5_: ", p_5_.shape, type(p_5_)) if print_shape_type else None

        # edge learning
        e_1 = self.e_1(x_2); print("e_1: ", e_1.shape, type(e_1)) if print_shape_type else None
        e_1_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(e_1, axis=-1),e_1.shape[-1]),-1); print("e_1_: ", e_1_.shape, type(e_1_)) if print_shape_type else None
        
        e_2 = self.e_2(p[0]); print("e_2: ", e_2.shape, type(e_2)) if print_shape_type else None
        e_2_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(e_2, axis=-1),e_2.shape[-1]),-1); print("e_2_: ", e_2_.shape, type(e_2_)) if print_shape_type else None
        
        e_3 = self.e_3(p[1]); print("e_3: ", e_3.shape, type(e_3)) if print_shape_type else None
        e_3_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(e_3, axis=-1),e_3.shape[-1]),-1); print("e_3_: ", e_3_.shape, type(e_3_)) if print_shape_type else None
        
        e_4 = self.e_4(p[2]); print("e_4: ", e_4.shape, type(e_4)) if print_shape_type else None
        e_4_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(e_4, axis=-1),e_4.shape[-1]),-1); print("e_4_: ", e_4_.shape, type(e_4_)) if print_shape_type else None
        
        e_5 = self.conc([e_1, e_2, e_3, e_4]); print("e_5: ", e_5.shape, type(e_5)) if print_shape_type else None
        e_5 = self.econv(e_5); print("e_5: ", e_5.shape, type(e_5)) if print_shape_type else None
        e_5_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(e_5, axis=-1),e_5.shape[-1]),-1); print("e_5_: ", e_5_.shape, type(e_5_)) if print_shape_type else None

        # region learning
        r_1 = self.r_1(x_2); print("r_1: ", r_1.shape, type(r_1)) if print_shape_type else None
        r_1_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(r_1, axis=-1),r_1.shape[-1]),-1); print("r_1_: ", r_1_.shape, type(r_1_)) if print_shape_type else None
        
        r_2 = self.r_2(p[0]); print("r_2: ", r_2.shape, type(r_2)) if print_shape_type else None
        r_2_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(r_2, axis=-1),r_2.shape[-1]),-1); print("r_2_: ", r_2_.shape, type(r_2_)) if print_shape_type else None
        
        r_3 = self.r_3(p[1]); print("r_3: ", r_3.shape, type(r_3)) if print_shape_type else None
        r_3_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(r_3, axis=-1),r_3.shape[-1]),-1); print("r_3_: ", r_3_.shape, type(r_3_)) if print_shape_type else None
        
        r_4 = self.r_4(p[2]); print("r_4: ", r_4.shape, type(r_4)) if print_shape_type else None
        r_4_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(r_4, axis=-1),r_4.shape[-1]),-1); print("r_4_: ", r_4_.shape, type(r_4_)) if print_shape_type else None
        
        r_5 = self.conc([r_1, r_2, r_3, r_4]); print("r_5: ", r_5.shape, type(r_5)) if print_shape_type else None
        r_5 = self.rconv(r_5); print("r_5: ", r_5.shape, type(r_5)) if print_shape_type else None
        r_5_ = tf.expand_dims(tf.divide(tf.math.reduce_sum(r_5, axis=-1),r_5.shape[-1]),-1); print("r_5_: ", r_5_.shape, type(r_5_)) if print_shape_type else None


        # direction learning
        x_3 = self.conc([p_5, e_5, r_5]); print("x_3: ", x_3.shape, type(x_3)) if print_shape_type else None
        x_4 = self.conv2(x_3); print("x_4: ", x_4.shape, type(x_4)) if print_shape_type else None
        d = self.d(x_4); print("d: ", d.shape, type(d)) if print_shape_type else None

        return p_5_, e_1_, e_2_, e_3_, e_4_, e_5_, r_1_, r_2_, r_3_, r_4_, r_5_, d

# Loading example
input_layer = Input(shape=(512,512,3,))
x = topology_model()(input_layer, print_shape_type=True)
model = tf.keras.Model(inputs=input_layer, outputs=x)
model.summary(expand_nested=False)

# inp = tf.ones((1,512,512,8))
# d = DirectionBlock(40)(inp)
# print(type(d))

# base = tf.keras.applications.resnet50.ResNet50(input_shape=((512,512,3)), include_top=False, weights='imagenet')
# base_model = tf.keras.Model(base.input, base.layers[174].output)
# inp = tf.ones((1,512,512,3))
# y = base_model(inp)
# print(y.shape)
# print(len(base.layers))
# print(len(base_model.layers))