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

import math
import os
import numpy as np
import tensorflow as tf

pred_hour = 24
variable_num = 4
target_index = -2
base_filters = 128
batch_size = 32
epochs = 200
building_lst = ['AT_SFH', 'AT_COM', 'AT_MFH', 'CH_SFH', 'CH_COM', 'CH_MFH', 'DE_SFH', 'DE_COM', 'DE_MFH'] 

LAMBDA_RECONSTRUCTION = 100.0
LAMBDA_DETECTION = 10.0
LAMBDA_ADVERSARIAL = 1.0

def upsample(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding="same", kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def downsample(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding="same", kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def dilated(filters, size, dilation_rate):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, size, strides=1, padding="same", dilation_rate=dilation_rate, kernel_initializer=initializer, use_bias=False),
        tf.keras.layers.ReLU()
    ])
    return result

@tf.keras.utils.register_keras_serializable(package="Preprocessing")
class AbsoluteDifference(tf.keras.layers.Layer):
    def call(self, inputs):
        observed, reconstructed = inputs
        return tf.abs(observed - reconstructed)

def Generator(pred_hour, variable_num, strides, base):
    initializer = tf.random_normal_initializer(0., 0.02)

    context_input = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name="context_input")
    observed_input = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name="observed_input")
    missing_mask = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name="missing_mask")

    down_stack = [
        downsample(base, 3, (2, 2), apply_batchnorm=False),
        downsample(base * 2, 3, strides),
        dilated(base * 2, 3, (1, 1)),
        dilated(base * 2, 3, (2, 2)),
        dilated(base * 2, 3, (4, 4)),
        dilated(base * 2, 3, (8, 8))
    ]
    up_stack = [upsample(base * 2, 3, (1, 1)), upsample(base, 3, strides)]

    x = context_input
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    for up, skip in zip(up_stack, reversed(skips[:2])):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    reconstruction = tf.keras.layers.Conv2DTranspose(
        1, 3, strides=(2, 2), padding="same", kernel_initializer=initializer,
        activation="softplus", name="reconstruction"
    )(x)

    detection_features = tf.keras.layers.Conv2DTranspose(
        32, 3, strides=(2, 2), padding="same", kernel_initializer=initializer,
        use_bias=False, name="detection_upsampling"
    )(x)
    detection_features = tf.keras.layers.BatchNormalization()(detection_features)
    detection_features = tf.keras.layers.LeakyReLU()(detection_features)

    residual = AbsoluteDifference(name="blind_residual")([observed_input, reconstruction])
    detection_features = tf.keras.layers.Concatenate(name="detection_information")(
        [detection_features, observed_input, reconstruction, residual, missing_mask]
    )
    detection_features = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_initializer=initializer, use_bias=False)(detection_features)
    detection_features = tf.keras.layers.BatchNormalization()(detection_features)
    detection_features = tf.keras.layers.LeakyReLU()(detection_features)
    detection_features = tf.keras.layers.Conv2D(32, 3, padding="same", kernel_initializer=initializer)(detection_features)
    detection_features = tf.keras.layers.LeakyReLU()(detection_features)
    anomaly_probability = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="anomaly_probability")(detection_features)

    return tf.keras.Model(
        inputs=[context_input, observed_input, missing_mask],
        outputs=[reconstruction, anomaly_probability],
        name="condition_aware_preprocessing_network"
    )

def global_discriminator(pred_hour, variable_num, strides):
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name="input_image")
    tar = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name="target_image")
    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(32, 3, strides)(x)
    down2 = downsample(64, 3, strides)(down1)
    down3 = downsample(128, 3, strides)(down2)
    x = tf.keras.layers.ZeroPadding2D()(down3)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, kernel_initializer=initializer, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    output = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=[inp, tar], outputs=output)

def get_target_selector(variable_num, target_index, dtype=tf.float32):
    if target_index < 0:
        target_index += variable_num
    selector = tf.one_hot(target_index, variable_num, dtype=dtype)
    return tf.reshape(selector, [1, 1, variable_num, 1])

def expand_events(seeds, duration=3):
    return tf.nn.max_pool2d(seeds, ksize=[1, duration, 1, 1], strides=[1, 1, 1, 1], padding="SAME")

def make_corrupted_batch(clean, anomaly_seed_rate=0.03, missing_seed_rate=0.02):
    dtype = clean.dtype
    selector = get_target_selector(variable_num, target_index, dtype)
    batch = tf.shape(clean)[0]
    hours = tf.shape(clean)[1]
    point_shape = tf.stack([batch, hours, 1, 1])

    anomaly_seeds = tf.cast(tf.random.uniform(point_shape) < anomaly_seed_rate, dtype)
    anomaly_mask = expand_events(anomaly_seeds, duration=3) * selector

    missing_seeds = tf.cast(tf.random.uniform(point_shape) < missing_seed_rate, dtype)
    missing_mask = expand_events(missing_seeds, duration=3) * selector * (1. - anomaly_mask)

    load = clean[:, :, target_index:target_index + 1, :]
    load_std = tf.math.reduce_std(load, axis=1, keepdims=True) + 1e-4
    sign = tf.where(
        tf.random.uniform([batch, 1, 1, 1]) < 0.5,
        -tf.ones([batch, 1, 1, 1], dtype=dtype),
        tf.ones([batch, 1, 1, 1], dtype=dtype)
    )

    spike = clean + sign * tf.random.uniform([batch, 1, 1, 1], 2., 5., dtype=dtype) * load_std
    bias = clean + sign * tf.random.uniform([batch, 1, 1, 1], 0.5, 1.5, dtype=dtype) * load_std
    stuck = tf.concat([clean[:, :1], clean[:, :-1]], axis=1)
    ramp = tf.reshape(tf.linspace(tf.cast(0., dtype), tf.cast(1.5, dtype), hours), [1, hours, 1, 1])
    drift = clean + sign * ramp * load_std

    fault_type = tf.random.uniform([batch, 1, 1, 1], minval=0, maxval=4, dtype=tf.int32)
    faulty = tf.where(fault_type == 0, spike, bias)
    faulty = tf.where(fault_type == 2, stuck, faulty)
    faulty = tf.where(fault_type == 3, drift, faulty)

    observed = tf.where(anomaly_mask > 0., faulty, clean)
    observed = tf.where(missing_mask > 0., tf.zeros_like(observed), observed)
    return observed, missing_mask, anomaly_mask

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def masked_mean(values, mask):
    mask = tf.cast(mask, values.dtype)
    return tf.reduce_sum(values * mask) / (tf.reduce_sum(mask) + 1e-8)

def focal_detection_loss(labels, probabilities, valid_mask, alpha=0.75, gamma=2.):
    probabilities = tf.clip_by_value(probabilities, 1e-6, 1. - 1e-6)
    p_t = labels * probabilities + (1. - labels) * (1. - probabilities)
    alpha_t = labels * alpha + (1. - labels) * (1. - alpha)
    loss = -alpha_t * tf.pow(1. - p_t, gamma) * tf.math.log(p_t)
    return masked_mean(loss, valid_mask)

def discriminator_loss(real_output, generated_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    generated_loss = loss_object(tf.zeros_like(generated_output), generated_output)
    return real_loss + generated_loss

def make_dataset(x, batch_size):
    return (
        tf.data.Dataset.from_tensor_slices(x)
        .shuffle(len(x), reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .repeat()
        .map(lambda batch: tf.cast(batch, tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

def build_train_step(generator, discriminator, optimizer_g, optimizer_d):
    @tf.function
    def train_step(x_batch):
        x_batch = tf.cast(x_batch, tf.float32)
        observed, missing_mask, anomaly_labels = make_corrupted_batch(x_batch)
        selector = get_target_selector(variable_num, target_index, x_batch.dtype)
        target_mask = tf.ones_like(x_batch) * selector

        # Hide the entire load column from the reconstruction pathway.
        gen_input = observed * (1. - selector)

        with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
            reconstruction, anomaly_probability = generator(
                [gen_input, observed, missing_mask], training=True
            )
            generated_full = observed * (1. - selector) + reconstruction * selector

            disc_real_output = discriminator([gen_input, x_batch], training=True)
            disc_generated_output = discriminator([gen_input, generated_full], training=True)

            reconstruction_loss = masked_mean(tf.abs(x_batch - reconstruction), target_mask)
            detection_valid_mask = target_mask * (1. - missing_mask)
            detection_loss = focal_detection_loss(anomaly_labels, anomaly_probability, detection_valid_mask)
            adversarial_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

            total_g_loss = (
                LAMBDA_RECONSTRUCTION * reconstruction_loss
                + LAMBDA_DETECTION * detection_loss
                + LAMBDA_ADVERSARIAL * adversarial_loss
            )
            total_d_loss = discriminator_loss(disc_real_output, disc_generated_output)

        grads_g = tape_g.gradient(total_g_loss, generator.trainable_variables)
        grads_d = tape_d.gradient(total_d_loss, discriminator.trainable_variables)
        optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
        optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))

        return total_g_loss, reconstruction_loss, detection_loss, adversarial_loss, total_d_loss
    return train_step

@tf.function
def preprocess_batch(generator, raw_batch, residual_threshold):
    raw_batch = tf.cast(raw_batch, tf.float32)
    selector = get_target_selector(
        variable_num, target_index, raw_batch.dtype
    )

    # Identify missing target-load observations.
    missing_mask = (
        tf.cast(tf.math.is_nan(raw_batch), raw_batch.dtype)
        * selector
    )
    observed = tf.where(
        tf.math.is_nan(raw_batch),
        tf.zeros_like(raw_batch),
        raw_batch
    )

    # Hide the entire target-load column from reconstruction.
    gen_input = observed * (1. - selector)

    reconstruction, anomaly_probability = generator(
        [gen_input, observed, missing_mask],
        training=False
    )

    # Compute an absolute difference separately at every position.
    absolute_difference = tf.abs(observed - reconstruction)

    # Detect only observed points in the target-load column.
    valid_mask = selector * (1. - missing_mask)
    detection_score = absolute_difference * valid_mask

    anomaly_mask = (
        tf.cast(
            absolute_difference > residual_threshold,
            raw_batch.dtype
        )
        * valid_mask
    )

    # Replace missing observations and detected anomalous points.
    replacement_mask = tf.clip_by_value(
        missing_mask + anomaly_mask, 0., 1.
    )
    cleaned_data = (
        observed * (1. - replacement_mask)
        + reconstruction * replacement_mask
    )

    return {
        "cleaned_data": cleaned_data,
        "reconstruction": reconstruction * selector,
        "detection_score": detection_score,

        # Retained for comparison; not used for anomaly decisions.
        "anomaly_probability": anomaly_probability * selector,

        "missing_mask": missing_mask,
        "anomaly_mask": anomaly_mask,
        "replacement_mask": replacement_mask
    }

for building in building_lst:
    x_train = np.load(f"/data/train_data/task_{building}_data_solar_24.npy")
    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]

    generator = Generator(pred_hour, variable_num, (2, 1), base_filters)
    discriminator = global_discriminator(pred_hour, variable_num, (2, 2))
    generator.summary()
    discriminator.summary()

    optimizer_g = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    optimizer_d = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    train_step = build_train_step(generator, discriminator, optimizer_g, optimizer_d)

    dataset = make_dataset(x_train, batch_size)
    ds_iter = iter(dataset)
    steps_per_epoch = math.ceil(len(x_train) / batch_size)
    os.makedirs("pre_trained_model", exist_ok=True)

    for epoch in range(epochs + 1):
        epoch_losses = np.zeros(5, dtype=np.float64)

        for _ in range(steps_per_epoch):
            losses = train_step(next(ds_iter))
            epoch_losses += np.array([float(loss) for loss in losses])

        epoch_losses /= steps_per_epoch

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: G={epoch_losses[0]:.4f}, "
                f"Rec={epoch_losses[1]:.4f}, Det={epoch_losses[2]:.4f}, "
                f"Adv={epoch_losses[3]:.4f}, D={epoch_losses[4]:.4f}"
            )
            generator.save(f"preprocessor_{building}.keras")

pred_hour, length, windows_length = 24, 24 * 30, 24
variable_num, temporal_num = 4, 1
target_index = variable_num - temporal_num - 1
normal_error_quantile = 0.90
random_anomaly_start_idx = 132
random_anomaly_end_idx = 132 + 24 * 20
sensor_failure_types = ['stuck', 'zero', 'bias', 'drift', 'noise']
failure_magnitude, spike_interval = 0.25, 6
test_lst = building_lst

SAVE_DIR = '/data/train_data/proposed_realtime_detection_npy'
RESULT_CSV = '/data/train_data/proposed_realtime_detection.csv'
RUNTIME_CSV = '/data/train_data/proposed_realtime_runtime.csv'
METRIC_COLUMNS = ['building', 'failure_type', 'nrmse', 'specificity', 'sensitivity', 'auc', 'f1-score', 'accuracy']
PLOT_RESULTS = True
os.makedirs(SAVE_DIR, exist_ok=True)

@tf.keras.utils.register_keras_serializable(package='Preprocessing')
class AbsoluteDifference(tf.keras.layers.Layer):
    def call(self, inputs):
        observed, reconstruction = inputs
        return tf.abs(observed - reconstruction)

def get_target_selector(variable_num, target_index, dtype=tf.float32):
    if target_index < 0: target_index += variable_num
    selector = tf.one_hot(target_index, variable_num, dtype=dtype)
    return tf.reshape(selector, [1, 1, variable_num, 1])

@tf.function
def preprocess_batch(generator, raw_batch, residual_threshold, residual_bias=0.0):
    raw_batch = tf.cast(raw_batch, tf.float32)
    selector = get_target_selector(variable_num, target_index, raw_batch.dtype)
    missing_mask = tf.cast(tf.math.is_nan(raw_batch), raw_batch.dtype) * selector
    observed = tf.where(tf.math.is_nan(raw_batch), tf.zeros_like(raw_batch), raw_batch)

    gen_input = observed * (1. - selector)
    reconstruction, anomaly_probability = generator([gen_input, observed, missing_mask], training=False)

    absolute_difference = tf.abs(observed - reconstruction - tf.cast(residual_bias, raw_batch.dtype) * selector)
    valid_mask = selector * (1. - missing_mask)
    detection_score = absolute_difference * valid_mask
    anomaly_mask = tf.cast(absolute_difference > residual_threshold, raw_batch.dtype) * valid_mask

    replacement_mask = tf.clip_by_value(missing_mask + anomaly_mask, 0., 1.)
    cleaned_data = observed * (1. - replacement_mask) + reconstruction * replacement_mask

    return {
        'cleaned_data': cleaned_data,
        'reconstruction': reconstruction * selector,
        'detection_score': detection_score,
        'anomaly_probability': anomaly_probability * selector,
        'missing_mask': missing_mask,
        'anomaly_mask': anomaly_mask,
        'replacement_mask': replacement_mask
    }

def calculate_nrmse(original, reconstructed):
    valid = np.isfinite(original) & np.isfinite(reconstructed)
    if not valid.any(): return np.nan
    original, reconstructed = original[valid], reconstructed[valid]
    data_range = np.ptp(original)
    return np.sqrt(np.mean((reconstructed - original) ** 2)) / data_range if data_range > 0 else np.nan

def test(building, failure_type='stuck'):
    generator = tf.keras.models.load_model(
        f'preprocessor_{building}.keras',
        custom_objects={'AbsoluteDifference': AbsoluteDifference, 'Preprocessing>AbsoluteDifference': AbsoluteDifference},
        compile=False
    )

    original_data = np.load(f'/data/train_data/task_{building}_data_solar_test_24.npy')[0:30] #[181:214]
    
    anomalous_data = original_data.copy().reshape(-1, variable_num, 1)
    failure_start = random_anomaly_start_idx
    failure_end = min(random_anomaly_end_idx, len(anomalous_data))

    true_anomaly_masks = np.zeros(len(anomalous_data), dtype=bool)
    true_anomaly_masks[failure_start:failure_end] = True
    segment = anomalous_data[failure_start:failure_end, target_index, 0]
    rng = np.random.default_rng(42)

    if failure_type == 'stuck':
        segment[:] = anomalous_data[failure_start - 1, target_index, 0]
    elif failure_type == 'zero':
        segment[:] = 0.0
    elif failure_type == 'bias':
        segment[:] += failure_magnitude
    elif failure_type == 'drift':
        segment[:] += np.linspace(0.0, failure_magnitude * 2, len(segment) + 1)[1:]
    elif failure_type == 'noise':
        segment[:] += rng.normal(0.0, failure_magnitude, len(segment))
    else:
        raise ValueError(f'Unknown sensor failure type: {failure_type}')

    anomalous_data = anomalous_data.reshape(original_data.shape)
    selector = np.zeros((1, 1, variable_num, 1), dtype=np.float32)
    selector[:, :, target_index, :] = 1.0

    original_lst, anomalous_lst, reconstructed_lst = [], [], []
    recovered_lst, anomaly_masks, detection_scores = [], [], []
    probability_lst, replacement_masks, missing_masks = [], [], []
    processing_times = []

    number_of_windows = min(len(anomalous_data), (length - pred_hour) // windows_length + 1)

    reference_batch = anomalous_data[:3].astype(np.float32)
    reference_missing = np.isnan(reference_batch).astype(np.float32) * selector
    reference_observed = np.where(np.isnan(reference_batch), 0.0, reference_batch)

    reference_reconstruction, _ = generator(
        [reference_observed * (1.0 - selector), reference_observed, reference_missing],
        training=False
    )

    reference_residual = (
        reference_batch[:, :, target_index, 0]
        - reference_reconstruction.numpy()[:, :, target_index, 0]
    )
    reference_residual = reference_residual[np.isfinite(reference_residual)]
    if not reference_residual.size:
        raise ValueError('No finite observations in the initial normal reference.')

    residual_bias = float(np.median(reference_residual))
    residual_threshold = max(float(np.quantile(
        np.abs(reference_residual - residual_bias),
        normal_error_quantile, method='higher'
    )), 1e-6)

    print(f'{building} | {failure_type} | Calibration=72h | '
          f'Residual bias={residual_bias:.6f} | Threshold={residual_threshold:.6f}')

    # Real-time inference: each 24-hour window is processed independently.
    for i in range(number_of_windows):
        raw_batch = anomalous_data[i:i + 1].astype(np.float32)

        start_time = time.perf_counter()
        result = preprocess_batch(
            generator, raw_batch,
            residual_threshold=residual_threshold,
            residual_bias=residual_bias
        )

        reconstruction = result['reconstruction'].numpy()
        anomaly_mask = result['anomaly_mask'].numpy()
        cleaned_data = result['cleaned_data'].numpy()
        detection_score = result['detection_score'].numpy()
        anomaly_probability = result['anomaly_probability'].numpy()
        replacement_mask = result['replacement_mask'].numpy()
        missing_mask = result['missing_mask'].numpy()
        processing_times.append(time.perf_counter() - start_time)

        original = original_data[i, :, target_index, 0]
        anomalous = raw_batch[0, :, target_index, 0]
        reconstructed = reconstruction[0, :, target_index, 0]
        recovered = cleaned_data[0, :, target_index, 0]

        original_lst.append(original.copy())
        anomalous_lst.append(anomalous.copy())
        reconstructed_lst.append(reconstructed.copy())
        recovered_lst.append(recovered.copy())
        anomaly_masks.append(anomaly_mask[0, :, target_index, 0].astype(bool))
        detection_scores.append(detection_score[0, :, target_index, 0].copy())
        probability_lst.append(anomaly_probability[0, :, target_index, 0].copy())
        replacement_masks.append(replacement_mask[0, :, target_index, 0].astype(bool))
        missing_masks.append(missing_mask[0, :, target_index, 0].astype(bool))

    # Combine the 30 independent outputs chronologically; calculate metrics once.
    np_original = np.stack(original_lst)
    np_anomalous = np.stack(anomalous_lst)
    np_reconstructed = np.stack(reconstructed_lst)
    np_recovered = np.stack(recovered_lst)
    anomaly_masks = np.stack(anomaly_masks)
    detection_scores = np.stack(detection_scores)
    anomaly_probabilities = np.stack(probability_lst)
    replacement_masks = np.stack(replacement_masks)
    missing_masks = np.stack(missing_masks)
    true_anomaly_masks = true_anomaly_masks.reshape(number_of_windows, pred_hour)

    expected = np.zeros(length, dtype=bool)
    expected[failure_start:failure_end] = True
    np.testing.assert_array_equal(true_anomaly_masks.reshape(-1), expected)

    if any(a.shape != (number_of_windows, pred_hour) for a in (
        np_original, np_anomalous, np_reconstructed, np_recovered,
        anomaly_masks, detection_scores, true_anomaly_masks
    )):
        raise ValueError('Unaligned hourly predictions or ground-truth labels.')

    hours = np.arange(number_of_windows * pred_hour).reshape(number_of_windows, pred_hour)
    recovery_mask = true_anomaly_masks & np.isfinite(np_original) & np.isfinite(np_reconstructed)
    nrmse = calculate_nrmse(np_original[recovery_mask], np_reconstructed[recovery_mask])

    evaluation_mask = (
        np.isfinite(np_original) & np.isfinite(np_anomalous)
        & np.isfinite(np_reconstructed) & np.isfinite(detection_scores)
    )
    if not np.all(evaluation_mask[true_anomaly_masks]):
        raise ValueError('Some injected fault hours have invalid values; detection coverage would be incomplete.')

    y_true = true_anomaly_masks[evaluation_mask]
    y_pred = anomaly_masks[evaluation_mask]
    y_score = detection_scores[evaluation_mask]

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    spe = tn / (tn + fp) if tn + fp else np.nan
    sen = tp / (tp + fn) if tp + fn else np.nan
    auc = float(roc_auc_score(y_true.astype(int), y_score)) if y_true.any() and (~y_true).any() else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    accuracy = (tp + tn) / len(y_true) if len(y_true) else np.nan

    np.testing.assert_array_equal(
        anomaly_masks,
        (detection_scores > residual_threshold) & (~missing_masks)
    )
    np.testing.assert_allclose(
        np_recovered[replacement_masks], np_reconstructed[replacement_masks],
        rtol=1e-5, atol=1e-6
    )
    retained = (~replacement_masks) & np.isfinite(np_anomalous)
    np.testing.assert_allclose(np_recovered[retained], np_anomalous[retained], rtol=1e-5, atol=1e-6)

    metrics = [{
        'building': building, 'failure_type': failure_type, 'nrmse': nrmse,
        'specificity': spe, 'sensitivity': sen, 'auc': auc,
        'f1-score': f1, 'accuracy': accuracy
    }]

    records = [{
        'building': building, 'failure_type': failure_type,
        'mean_processing_time_s': float(np.mean(processing_times)),
        'max_processing_time_s': float(np.max(processing_times)),
        'total_processing_time_s': float(np.sum(processing_times)),
        'within_24h_deadline': bool(np.all(np.asarray(processing_times) <= pred_hour * 3600))
    }]

    # Save the complete result, retaining the 30 x 24 structure.
    saved_data = {
        'hour': hours,
        'original': np_original,
        'corrupted': np_anomalous,
        'reference': np_reconstructed,
        'recovered': np_recovered,
        'true_anomaly': true_anomaly_masks,
        'detected_anomaly': anomaly_masks,
        'anomaly_score': detection_scores,
        'anomaly_probability': anomaly_probabilities,
        'replacement_mask': replacement_masks,
        'missing_mask': missing_masks,
        'residual_bias': np.float64(residual_bias),
        'residual_threshold': np.float64(residual_threshold),
        'residual_quantile': np.float64(normal_error_quantile),
        'calibration_hours': np.int64(3 * pred_hour)
    }
    filename = f'{building}_{failure_type}_Proposed-Residual_warm_month.npy'
    np.save(os.path.join(SAVE_DIR, filename), saved_data)

    print(f'Failure={failure_type}, NRMSE (failure interval)={nrmse:.6f}')
    print(f'Detection samples={len(y_true)}, Actual anomalies={tp+fn}, Actual normal={tn+fp}')
    print(f'TP={tp}, FP={fp}, FN={fn}, TN={tn}')
    print(f'Specificity={spe:.4f}, Sensitivity={sen:.4f}, AUC={auc:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}')
    print(f'Saved: {os.path.join(SAVE_DIR, filename)}')

    if PLOT_RESULTS:
        flat_hours = hours.reshape(-1)
        original_flat = np_original.reshape(-1)
        anomalous_flat = np_anomalous.reshape(-1)
        reconstructed_flat = np_reconstructed.reshape(-1)
        recovered_flat = np_recovered.reshape(-1)
        true_flat = true_anomaly_masks.reshape(-1)
        detected_flat = anomaly_masks.reshape(-1)

        for start, end in [(0, len(flat_hours)), (24 * 20, min(24 * 30, len(flat_hours)))]:
            if start >= end: continue
            local_hours = flat_hours[start:end]
            local_anomalous = anomalous_flat[start:end]
            local_true = true_flat[start:end]
            local_detected = detected_flat[start:end]

            plt.figure(figsize=(14, 5))
            plt.plot(local_hours, original_flat[start:end], label='Original', color='green', linewidth=2)
            plt.plot(local_hours, reconstructed_flat[start:end], label='Reconstructed', color='blue', alpha=0.8)
            plt.plot(local_hours, recovered_flat[start:end], label='Recovered', color='purple', alpha=0.8)
            plt.plot(local_hours, local_anomalous, label='After adding anomalies', color='red', linestyle='--')
            plt.scatter(local_hours[local_true], local_anomalous[local_true], label='Injected anomaly',
                        facecolors='none', edgecolors='darkorange', s=45, zorder=4)
            plt.scatter(local_hours[local_detected], local_anomalous[local_detected], label='Detected anomaly',
                        color='black', marker='x', s=25, zorder=5)
            plt.title(f'Building {building} | {failure_type} | SPE={spe:.3f}, SEN={sen:.3f}, F1={f1:.3f}')
            plt.xlabel('Hour')
            plt.ylabel('Normalised load')
            plt.legend()
            plt.tight_layout()
            plt.show()

    del generator
    tf.keras.backend.clear_session()
    return pd.DataFrame(records), pd.DataFrame(metrics, columns=METRIC_COLUMNS)

all_records = []
all_metrics = []
for building in building_lst:
    for failure_type in sensor_failure_types:
        building_records, building_metrics = test(building, failure_type)
        all_records.append(building_records)
        all_metrics.append(building_metrics)

pd_records = pd.concat(all_records, ignore_index=True)
pd_metrics = pd.concat(all_metrics, ignore_index=True)
pd_records.to_csv(RUNTIME_CSV, index=False)
pd_metrics[METRIC_COLUMNS].to_csv(RESULT_CSV, index=False)
print(f'Saved {len(pd_metrics)} whole-series results to {RESULT_CSV}')
print(pd_metrics[METRIC_COLUMNS].to_string(index=False))
