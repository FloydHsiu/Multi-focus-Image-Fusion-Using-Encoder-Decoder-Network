import tensorflow as tf
from data import parseDataset
import model
import datetime
from os import path
import tqdm
import argparse


@tf.function(experimental_relax_shapes=True)
def convertData(A, B, label):
    A = A / 255.0
    B = B / 255.0
    A = A * 2.0 - 1.0
    B = B * 2.0 - 1.0
    label = label * 2.0 - 1.0
    return A, B, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--tpu_name', dest='tpu_name', type=str, default='',
        help='Assign tpu that you want to train this code on')
    parser.add_argument(
        '--data_path', type=str, default='',
        help='Assign your training data (tfrecords)')
    parser.add_argument(
        '--logs_dir', tpye=str, default='',
        help='Assisn your directory to keep training logs')
    parser.add_argument(
        '--lytro_dir', type=str, default='',
        help='Assign directory of Lytro Multi-focus Dataset')

    args = parser.parse_args()

    if parser.tpu_name == '':
        print('Error: have no tpu_name been declared.')
        return
    if parser.data_path == '':
        print('Error: have no training data path been declared.')
        return
    if parser.logs_dir == '':
        print('Error: have no training logs directory been declared')
        return
    if parser.lytro_dir == '':
        print('Error: have no lytro multi-focus dataset directory been declared')

    batch_size = 128
    learning_rate = 1e-4
    learning_rate_decay = 1e-1
    epoch = 2

    TPU_NAME = parser.tpu_name
    DATA_PATH = parser.data_path
    LOGS_DIR = parser.logs_dir
    LYTRO_DIR = parser.lytro_dir

    # TPU distributed computation initialization
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=TPU_NAME)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

    ############### DATAs ###############
    # split training data and valiation data
    dataset = tf.data.TFRecordDataset(DATA_PATH)
    dataset = dataset.map(parseDataset).shuffle(
        buffer_size=10000, reshuffle_each_iteration=False)

    train_dataset = dataset.take(90000).batch(batch_size, drop_remainder=False)
    test_dataset = dataset.skip(90000).batch(batch_size, drop_remainder=False)

    train_dist_dataset = tpu_strategy.experimental_distribute_dataset(
        train_dataset)
    test_dist_dataset = tpu_strategy.experimental_distribute_dataset(
        test_dataset)

    with tpu_strategy.scope():
        mfnet = model.MFNet()

    ############### LOGs ###############
    dt_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"############ {dt_now} ############")
    tensorboard_log_dir = path.join(LOGS_DIR, 'fcn_origin_'+dt_now)

    # checkpoint initialize
    checkpoint_dir = path.join(tensorboard_log_dir, 'training_checkpoints')
    makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(mfnet=mfnet)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=5)

    # Model directory initialize
    model_dir = path.join(tensorboard_log_dir, 'model')

    # tensorboard initialize
    tensorboard_dir = path.join(tensorboard_log_dir, 'tensorboard')
    makedirs(tensorboard_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)

    # Read Images for Inference
    fns = ["lytro-03-A.jpg", "lytro-03-B.jpg",
           "lytro-05-A.jpg", "lytro-05-B.jpg"]
    paths = [path.join(LYTRO_DIR, fn) for fn in fns]

    imgs = []
    for i in range(0, len(paths), 2):
        tmp1 = tf.io.read_file(paths[i])
        tmp2 = tf.io.read_file(paths[i+1])
        img1 = tf.cast(
            tf.io.decode_jpeg(tmp1, channels=1),
            tf.float32) / 255.0 * 2.0 - 1
        img2 = tf.cast(
            tf.io.decode_jpeg(tmp2, channels=1),
            tf.float32) / 255.0 * 2.0 - 1
        img1 = tf.reshape(img1, (1, 520, 520, 1))
        img2 = tf.reshape(img2, (1, 520, 520, 1))
        img1 = tf.slice(img1, [0, 0, 0, 0], [1, 512, 512, 1])
        img2 = tf.slice(img2, [0, 0, 0, 0], [1, 512, 512, 1])
        imgs.append([img1, img2])

    # stage 1
    with tpu_strategy.scope():
        optimizer_1 = tf.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.5)
        optimizer_2 = tf.optimizers.Adam(
            learning_rate=learning_rate*learning_rate_decay, beta_1=0.5)

    @tf.function
    def validation(dist_inputs):
        def step_fn(inputs):
            p1 = inputs['p1']
            p2 = inputs['p2']
            label = inputs['label']
            p1, p2, label = convertData(p1, p2, label)
            pred = mfnet([p1, p2], training=False)
            loss_fn = tf.keras.losses.Huber(
                delta=0.2, reduction=tf.keras.losses.Reduction.NONE)
            mae = loss_fn(label, pred)
            loss = tf.reduce_sum(mae, keepdims=True) / (320 * 320)
            return loss
        per_example_losses = tpu_strategy.experimental_run_v2(
            step_fn, args=(dist_inputs,))
        mean_loss = tpu_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)

        return mean_loss

    @tf.function
    def train_step_1(dist_inputs):
        # In training step 1, learning rate is set as 1e-4
        def step_fn(inputs):
            p1 = inputs['p1']
            p2 = inputs['p2']
            label = inputs['label']
            p1, p2, label = convertData(p1, p2, label)
            with tf.GradientTape() as g_tape:
                pred = mfnet([p1, p2], training=True)
                loss_fn = tf.keras.losses.Huber(
                    delta=0.2, reduction=tf.keras.losses.Reduction.NONE)
                mae = loss_fn(label, pred)
                loss = tf.reduce_sum(
                    mae, keepdims=True) / (batch_size * 320 * 320)
            grad = g_tape.gradient(loss, mfnet.trainable_variables)
            optimizer_1.apply_gradients(
                list(zip(grad, mfnet.trainable_variables)))
            return loss
        per_example_losses = tpu_strategy.experimental_run_v2(
            step_fn, args=(dist_inputs,))
        mean_loss = tpu_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)

        return mean_loss

    @tf.function
    def train_step_2(dist_inputs):
        # In training step 1, learning rate is set as 1e-4 * 1e-1
        def step_fn(inputs):
            p1 = inputs['p1']
            p2 = inputs['p2']
            label = inputs['label']
            p1, p2, label = convertData(p1, p2, label)
            with tf.GradientTape() as g_tape:
                pred = mfnet([p1, p2], training=True)
                loss_fn = tf.keras.losses.Huber(
                    delta=0.2, reduction=tf.keras.losses.Reduction.NONE)
                mae = loss_fn(label, pred)
                loss = tf.reduce_sum(
                    mae, keepdims=True) / (batch_size * 320 * 320)
            grad = g_tape.gradient(loss, mfnet.trainable_variables)
            optimizer_2.apply_gradients(
                list(zip(grad, mfnet.trainable_variables)))
            return loss
        per_example_losses = tpu_strategy.experimental_run_v2(
            step_fn, args=(dist_inputs,))
        mean_loss = tpu_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)

        return mean_loss

    @tf.function
    def inference():
        result = []
        for i_s in imgs:
            img1 = i_s[0]
            img2 = i_s[1]
            alpha = mfnet([img1, img2], training=False)
            alpha = (alpha+1.0)/2.0
            result.append(alpha)
        return result

    train_step = train_step_1

    i = 0
    for e in range(epoch):
        if e == epoch//2:
            train_step = train_step_2
        for inputs in tqdm.tqdm(train_dist_dataset):
            with tpu_strategy.scope():
                loss = train_step(inputs)

            if i % 1000 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
            if i % 100 == 0:
                total_loss = 0.0
                count = 10000
                for inputs_val in test_dist_dataset:
                    with tpu_strategy.scope():
                        val_loss = tf.squeeze(validation(inputs_val))
                        total_loss += val_loss
                with summary_writer.as_default():
                    tf.summary.scalar('val', total_loss/count, step=i)
                    result = inference()
                    for j in range(len(result)):
                        tf.summary.image(
                            f"Test Image {j+1}", result[j],
                            step=i)
            with summary_writer.as_default():
                tf.summary.scalar('loss', tf.squeeze(loss), step=i)
            i = i + 1
        tf.saved_model.save(mfnet, model_dir)
    tf.saved_model.save(mfnet, model_dir)
    checkpoint.save(file_prefix=checkpoint_prefix)
