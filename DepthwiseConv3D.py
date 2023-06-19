# -*- coding: utf-8 -*-

'''
This is a modification of the SeparableConv3D code in Keras,
to perform just the Depthwise Convolution (1st step) of the
Depthwise Separable Convolution layer.
'''
from __future__ import absolute_import
import os
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import layers
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils
#from keras.legacy.interfaces import conv3d_args_preprocessor, generate_legacy_interface
from tensorflow.keras.layers import Conv3D
import tensorflow as tf
#from keras.backend.tensorflow_backend import _preprocess_padding, _preprocess_conv3d_input
from tensorflow.python.client import device_lib

import tensorflow as tf

_MANUAL_VAR_INIT = False

def manual_variable_initialization(value):
    """Sets the manual variable initialization flag.

    This boolean flag determines whether
    variables should be initialized
    as they are instantiated (default), or if
    the user should handle the initialization
    (e.g. via `tf.initialize_all_variables()`).

    # Arguments
        value: Python boolean.
    """
    global _MANUAL_VAR_INIT
    _MANUAL_VAR_INIT = value


def _preprocess_padding(padding):
    """Convert keras' padding to tensorflow's padding.

    # Arguments
        padding: string, `"same"` or `"valid"`.

    # Returns
        a string, `"SAME"` or `"VALID"`.

    # Raises
        ValueError: if `padding` is invalid.
    """
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid padding: ' + str(padding))
    return padding
class _TfDeviceCaptureOp(object):
    """Class for capturing the TF device scope."""

    def __init__(self):
        self.device = None

    def _set_device(self, device):
        """This method captures TF's explicit device scope setting."""
        self.device = device

def _get_current_tf_device():
    """Return explicit device of current context, otherwise returns `None`.

    # Returns
        If the current device scope is explicitly set, it returns a string with
        the device (`CPU` or `GPU`). If the scope is not explicitly set, it will
        return `None`.
    """
    g = tf.get_default_graph()
    op = _TfDeviceCaptureOp()
    g._apply_device_functions(op)
    return op.device
def get_session():
    """Returns the TF session to be used by the backend.

    If a default TensorFlow session is available, we will return it.

    Else, we will return the global Keras session.

    If no global Keras session exists at this point:
    we will create a new global session.

    Note that you can manually set the global session
    via `K.set_session(sess)`.

    # Returns
        A TensorFlow session.
    """
    global _SESSION

    default_session = tf.get_default_session()

    if default_session is not None:
        session = default_session
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                num_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                        allow_soft_placement=True)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        with session.graph.as_default():
            variables = tf.global_variables()
            candidate_vars = []
            for v in variables:
                if not getattr(v, '_keras_initialized', False):
                    candidate_vars.append(v)
            if candidate_vars:
                # This step is expensive, so we only run it on variables
                # not already marked as initialized.
                is_initialized = session.run(
                    [tf.is_variable_initialized(v) for v in candidate_vars])
                uninitialized_vars = []
                for flag, v in zip(is_initialized, candidate_vars):
                    if not flag:
                        uninitialized_vars.append(v)
                    v._keras_initialized = True
                if uninitialized_vars:
                    session.run(tf.variables_initializer(uninitialized_vars))
    # hack for list_devices() function.
    # list_devices() function is not available under tensorflow r1.3.
    if not hasattr(session, 'list_devices'):
        session.list_devices = lambda: device_lib.list_local_devices()
    return session


def _is_current_explicit_device(device_type):
    """Check if the current device is explicitly set on the device type specified.

    # Arguments
        device_type: A string containing `GPU` or `CPU` (case-insensitive).

    # Returns
        A boolean indicating if the current device scope is explicitly set on the device type.

    # Raises
        ValueError: If the `device_type` string indicates an unsupported device.
    """
    device_type = device_type.upper()
    if device_type not in ['CPU', 'GPU']:
        raise ValueError('`device_type` should be either "CPU" or "GPU".')
    device = _get_current_tf_device()
    return (device is not None and device.device_type == device_type.upper())


def get_session():
    """Returns the TF session to be used by the backend.

    If a default TensorFlow session is available, we will return it.

    Else, we will return the global Keras session.

    If no global Keras session exists at this point:
    we will create a new global session.

    Note that you can manually set the global session
    via `K.set_session(sess)`.

    # Returns
        A TensorFlow session.
    """
    global _SESSION

    default_session = tf.get_default_session()

    if default_session is not None:
        session = default_session
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                num_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                        allow_soft_placement=True)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        with session.graph.as_default():
            variables = tf.global_variables()
            candidate_vars = []
            for v in variables:
                if not getattr(v, '_keras_initialized', False):
                    candidate_vars.append(v)
            if candidate_vars:
                # This step is expensive, so we only run it on variables
                # not already marked as initialized.
                is_initialized = session.run(
                    [tf.is_variable_initialized(v) for v in candidate_vars])
                uninitialized_vars = []
                for flag, v in zip(is_initialized, candidate_vars):
                    if not flag:
                        uninitialized_vars.append(v)
                    v._keras_initialized = True
                if uninitialized_vars:
                    session.run(tf.variables_initializer(uninitialized_vars))
    # hack for list_devices() function.
    # list_devices() function is not available under tensorflow r1.3.
    if not hasattr(session, 'list_devices'):
        session.list_devices = lambda: device_lib.list_local_devices()
    return session


def _has_nchw_support():
    """Check whether the current scope supports NCHW ops.

    TensorFlow does not support NCHW on CPU. Therefore we check if we are not explicitly put on
    CPU, and have GPUs available. In this case there will be soft-placing on the GPU device.

    # Returns
        bool: if the current scope device placement would support nchw
    """
    explicitly_on_cpu = _is_current_explicit_device('CPU')
    gpus_available = len(_get_available_gpus()) > 0
    return (not explicitly_on_cpu and gpus_available)

def _preprocess_conv3d_input(x, data_format):
    """Transpose and cast the input before the conv3d.

    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.
    """

    tf_data_format = 'NDHWC'
    if data_format == 'channels_first':
        if not _has_nchw_support():
            x = tf.transpose(x, (0, 2, 3, 4, 1))
        else:
            tf_data_format = 'NCDHW'
    return x, tf_data_format
def depthwise_conv3d_args_preprocessor(args, kwargs):
    converted = []

    if 'init' in kwargs:
        init = kwargs.pop('init')
        kwargs['depthwise_initializer'] = init
        converted.append(('init', 'depthwise_initializer'))

    args, kwargs, _converted = conv3d_args_preprocessor(args, kwargs)
    return args, kwargs, converted + _converted

    legacy_depthwise_conv3d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=depthwise_conv3d_args_preprocessor)


class DepthwiseConv3D(Conv3D):
    """Depthwise 3D convolution.
    Depth-wise part of separable convolutions consist in performing
    just the first step/operation
    (which acts on each input channel separately).
    It does not perform the pointwise convolution (second step).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.
    # Arguments
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along the depth, width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filterss_in * depth_multiplier`.
        groups: The depth size of the convolution (as a variant of the original Depthwise conv)
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        dialation_rate: List of ints.
                        Defines the dilation factor for each dimension in the
                        input. Defaults to (1,1,1)
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        5D tensor with shape:
        `(batch, depth, channels, rows, cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, depth, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(batch, filters * depth, new_depth, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_depth, new_rows, new_cols, filters * depth)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    #@legacy_depthwise_conv3d_support
    def __init__(self,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='same',
                 depth_multiplier=1,
                 groups=None,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 dilation_rate = (1, 1, 1),
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv3D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            dilation_rate=dilation_rate,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.groups = groups
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.dilation_rate = dilation_rate
        self._padding = _preprocess_padding(self.padding)
        self._strides = (1,) + self.strides + (1,)
        self._data_format = "NDHWC"
        self.input_dim = None

    def build(self, input_shape):
        if len(input_shape) < 5:
            raise ValueError('Inputs to `DepthwiseConv3D` should have rank 5. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv3D` '
                             'should be defined. Found `None`.')
        self.input_dim = int(input_shape[channel_axis])

        if self.groups is None:
            self.groups = self.input_dim

        if self.groups > self.input_dim:
            raise ValueError('The number of groups cannot exceed the number of channels')

        if self.input_dim % self.groups != 0:
            raise ValueError('Warning! The channels dimension is not divisible by the group size chosen')

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  self.kernel_size[2],
                                  self.input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.groups * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs, training=None):
        inputs = _preprocess_conv3d_input(inputs, self.data_format)

        if self.data_format == 'channels_last':
            dilation = (1,) + self.dilation_rate + (1,)
        else:
            dilation = self.dilation_rate + (1,) + (1,)

        if self._data_format == 'NCDHW':
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, i:i+self.input_dim//self.groups, :, :, :], self.depthwise_kernel[:, :, :, i:i+self.input_dim//self.groups, :],
                    strides=self._strides,
                    padding=self._padding,
                    dilations=dilation,
                    data_format=self._data_format) for i in range(0, self.input_dim, self.input_dim//self.groups)], axis=1)

        else:
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, :, :, :, i:i+self.input_dim//self.groups], self.depthwise_kernel[:, :, :, i:i+self.input_dim//self.groups, :],
                    strides=self._strides,
                    padding=self._padding,
                    dilations=dilation,
                    data_format=self._data_format) for i in range(0, self.input_dim, self.input_dim//self.groups)], axis=-1)

        if self.bias is not None:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            depth = input_shape[2]
            rows = input_shape[3]
            cols = input_shape[4]
            out_filters = self.groups * self.depth_multiplier
        elif self.data_format == 'channels_last':
            depth = input_shape[1]
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = self.groups * self.depth_multiplier

        depth = conv_utils.conv_output_length(depth, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])

        rows = conv_utils.conv_output_length(rows, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        cols = conv_utils.conv_output_length(cols, self.kernel_size[2],
                                             self.padding,
                                             self.strides[2])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, depth, rows, cols)

        elif self.data_format == 'channels_last':
            return (input_shape[0], depth, rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv3D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

DepthwiseConvolution3D = DepthwiseConv3D

