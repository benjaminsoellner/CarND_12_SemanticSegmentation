import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function # DONE?
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.save_model.loader.load(sess, [vgg_tag], vgg_path) # vgg_tag instead of path?
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: compare with https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    # see classroom: Scene Understanding / 8. FCN-8 Decoder
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    last_layer = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, padding='SAME', kernel_regularizer=regularizer)
    last_layer = tf.layers.conv2d_transpose(last_layer, num_classes, kernel_size=4, strides=(2, 2), padding='SAME', kernel_regularizer=regularizer)
    last_layer = tf.add(last_layer, vgg_layer4_out)
    last_layer = tf.layers.conv2d_transpose(last_layer, num_classes, kernel_size=4, strides=(2, 2), padding='SAME', kernel_regularizer=regularizer)
    last_layer = tf.add(last_layer, vgg_layer3_out)
    last_layer = tf.layers.conv2d_transpose(last_layer, num_classes, kernel_size=16, strides=(8, 8), padding='SAME', kernel_regularizer=regularizer)
    return last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # see classroom: check Scene Understanding / 9. FCN-8 Classification & Loss
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, correct_label))
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    return logits, training_operation, cross_entropy
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function # DONE?
    # TODO: implement Training (21:28 in project) - define loss with session.run(...) using optimizer from optimize(...) # DONE?
    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch+1))
        for input_image_data, correct_label_data in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: input_image_data, correct_label: correct_label_data,
                keep_prob: 0.75, learning_rate: 0.001 })
            print("Loss = {:.3f}".format(loss))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)


        # TODO: tune Hyperparams
        N = 3
        epochs = 5
        batch_size = 100
        # Placeholders
        x = tf.placeholder(tf.float32, (None, None, None, 3))
        y = tf.placeholder(tf.int32, (None, None, None, 1))
        learning_rate = tf.placeholder(tf.float32)

        # TODO: Build NN using load_vgg, layers, and optimize function, call optimizer # DONE?
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, training_operation, cross_entropy = optimize(last_layer, y, learning_rate, N)
        # TODO: Train NN using the train_nn function # DONE?
        train_nn(sess, epochs, batch_size, get_batches_fn, training_operation, cross_entropy, x, y, keep_prob, learning_rate)
        # TODO: Save inference data using helper.save_inference_samples # DONE?
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # TODO OPTIONALS
        # * Apply the trained model to a video
        # * Augment Images for better results
        #   https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # * Use 3 classes
        # * Use citiscape instead of kitti


if __name__ == '__main__':
    run()
