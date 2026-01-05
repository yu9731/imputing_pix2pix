import numpy as np
import tensorflow as tf
import datetime

import tensorflow as tf
import numpy as np

if (tf.test.is_gpu_available):
    print("GPU")
else:
    print("CPU")

# List all physical GPUs available to TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print("Error enabling memory growth:", e)

pred_hour = 24
variable_num = 4
temporal_num = 1

building_lst = ['AT_SFH', 'AT_COM', 'CH_SFH', 'CH_COM', 'DE_SFH', 'DE_COM']

for building in building_lst:
      x_train = np.load(f'/kaggle/input/when2heat-gf/task_{building}_data_solar_24.npy')

      def upsample(filters, size, strides, apply_batchnorm=True, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
      
        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False))
      
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
      
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
      
        result.add(tf.keras.layers.LeakyReLU())
        return result
      
      def downsample(filters, size, strides, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
      
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False))
      
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
      
        result.add(tf.keras.layers.LeakyReLU())
      
        return result
      
      def dilated(filters, size, dilation_rate):
          initializer = tf.random_normal_initializer(0., 0.02)
      
          result = tf.keras.Sequential()
          result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                                                     dilation_rate=dilation_rate,
                                                                     kernel_initializer=initializer,
                                                                     use_bias=False))
      
          result.add(tf.keras.layers.ReLU())
      
          return result
      
      def Generator(pred_hour, variable_num, strides):
        inputs = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1])
      
        down_stack = [
          downsample(128, 3, strides, apply_batchnorm=False),
          downsample(256, 3, (2,2)),   # z-score: , apply_batchnorm=False # elec: strides -> (2,1)
          dilated(256, 3, (1,1)),
          dilated(256, 3, (2,2)),
          dilated(256, 3, (4,4)),
          dilated(256, 3, (8,8)),
        ]
      
        up_stack = [
          upsample(256, 3, (1,1)),
          upsample(128, 3, (2,2)),
        ]
      
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(1, 3,
                                               strides=strides,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='softplus')
      
        x = inputs
      
        # Downsampling through the model
        skips = []
        for down in down_stack:
          x = down(x)
          skips.append(x)
      
        skips = reversed(skips[0:2])
      
        for up, skip in zip(up_stack, skips):
          x = up(x)
          x = tf.keras.layers.Concatenate()([x, skip]) #Connecting as an 'U' in U-Net
      
        x = last(x)
      
        return tf.keras.Model(inputs=inputs, outputs=x)
      
      generator = Generator(pred_hour, variable_num, 2)
      generator.summary()
      
      def Discriminator(pred_hour, strides):
          initializer = tf.random_normal_initializer(0., 0.02)
      
          inp = tf.keras.layers.Input(shape=[pred_hour, 4, 1], name='input_image')
          tar = tf.keras.layers.Input(shape=[pred_hour, 4, 1], name='target_image')
      
          x = tf.keras.layers.concatenate([inp, tar])
      
          down1 = downsample(32, 3, strides,False)(x)
          down2 = downsample(64, 3, strides)(down1)
          down3 = downsample(128, 3, strides)(down2)
      
          zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
          conv = tf.keras.layers.Conv2D(128, 3, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)
      
          batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
          leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
      
          zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
      
          last = tf.keras.layers.Conv2D(1, 3, strides=1,
                                        kernel_initializer=initializer)(zero_pad2)
      
          return tf.keras.Model(inputs=[inp, tar], outputs=last)
      
      discriminator = Discriminator(pred_hour, 2)
      discriminator.summary()
      
      def get_points(pred_hour, batch_size, variable_num):
        mask = []
        points = []
        for i in range(batch_size):
          m = np.zeros((pred_hour, variable_num, 1), dtype=np.uint8)
          x1 = np.random.randint(0, pred_hour - pred_hour + 1, 1)[0]
          x2 = x1 + pred_hour
          points.append([x1, x2])
      
          m[:, -2] = 1
          mask.append(m)
        return np.array(mask)
      
      loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
      
      def generator_loss(disc_generated_output, gen_output, target):
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
      
        l2_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (100 * l2_loss)
      
        return total_gen_loss
      
      def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
      
        return total_disc_loss
      
      def make_dataset(x, batch_size):
          ds = tf.data.Dataset.from_tensor_slices(x)
          ds = ds.shuffle(buffer_size=len(x))
          ds = ds.take(int(len(x))).batch(batch_size).repeat()
          return ds.map(lambda batch: tf.cast(batch,tf.float32),
                       num_parallel_calls=tf.data.AUTOTUNE)

      batch_size = 32
      dataset = make_dataset(x_train, batch_size = batch_size)
      ds_iter     = iter(dataset)
      steps_per_epoch = int(len(x_train) / batch_size)
      
      optimizer_g = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
      optimizer_d = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
      
      @tf.function
      def train_step(batch_tuple, batch_size, variable_num):
          masks = get_points(pred_hour, batch_size, variable_num)
          masks = tf.cast(masks, tf.float32)
          total_g_loss = 0
          total_d_loss = 0    
          with tf.GradientTape(persistent=True) as tape:
              x_batch = tf.cast(batch_tuple, tf.float32)
      
              gen_input = x_batch * (1 - masks)
              gen_output = generator(gen_input, training=True)
          
              disc_real_output = discriminator([tf.cast(gen_input, tf.float32), tf.cast(x_batch, tf.float32)], training=True)
              disc_generated_output = discriminator([gen_output, tf.cast(x_batch, tf.float32)], training=True)
      
              gen_total_loss = generator_loss(disc_generated_output, gen_output, x_batch)
              total_g_loss += gen_total_loss
      
              disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
              total_d_loss += disc_loss
      
          grads_g = tape.gradient(total_g_loss, generator.trainable_variables)
          optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
      
          grads_d = tape.gradient(total_d_loss, discriminator.trainable_variables)
          optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
      
          return total_g_loss, total_d_loss

      for epoch in range(200+1):
          for _ in range(steps_per_epoch):
              batch = next(ds_iter)
              bs = batch.shape[0]
              g_loss, d_loss = train_step(batch, bs, variable_num)
      
          if epoch % 10 == 0:
              print(f"Epoch {epoch:03d}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
              generator.save(f'pre_trained_model/generator_{building}.h5')
