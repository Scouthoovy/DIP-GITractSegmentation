from tensorflow.keras import layers, Model

#Convolutional block with two Conv2D layers, optional batch norm and dropout
def conv_block(inputs, n_filters, dropout=0.0, batch_norm=True):
  """
  Apply two 3x3 convolutions
  
  Parameters
    inputs (tf.Tensor): Input feature map
    n_filters (int): Number of convolutional filters
    dropout (float, optional): Dropout rate between 0 and 1, with default 0.0.
    batch_norm (boolean, optional): If true, include batch normalization layers, with default True.
    
  Returns
    x (tf.Tensor): Transformed feature map
  """
  x = layers.Conv2D(n_filters, 3, padding="same")(inputs)
  if batch_norm:
    x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  x = layers.Conv2D(n_filters, 3, padding="same")(x)
  if batch_norm:
    x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  if dropout:
    x = layers.Dropout(dropout)(x)

  return x

#Encoder block applies conv_block followed by max pooling
def encoder_block(inputs, n_filters, dropout=0.0, batch_norm=True):
  """
  Apply conv_block followed by 2x2 max pooling for encoding
  
  Parameters
    inputs (tf.Tensor): Input feature map
    n_filters (int): Number of convolutional filters
    dropout (float, optional): Dropout rate between 0 and 1, with default 0.0.
    batch_norm (boolean, optional): If true, include batch normalization layers, with default True.
    
  Returns
    x (tf.Tensor): Transformed feature map before pooling
    p (tf.Tensor): Pooled feature map
  """
  x = conv_block(inputs, n_filters, dropout, batch_norm)
  p = layers.MaxPooling2D((2, 2), padding='same')(x)
  return x, p

#Decoder block applies upsampling, concatenation with skips and then conv_blocks.
def decoder_block(inputs, skip_features, n_filters, dropout=0.0, batch_norm=True):
  """
  Upsamples and concatenates with skip connections, then apply conv_block for decoding
  
  Parameters
    inputs (tf.Tensor): Input feature map
    skip_features (tf.Tensor): Corresponding encoder feature map for skip connections
    n_filters (int): Number of convolutional filters
    dropout (float, optional): Dropout rate between 0 and 1, with default 0.0.
    batch_norm (boolean, optional): If true, include batch normalization layers, with default True.
  
  Returns
    x (tf.Tensor): Decoded feature map
  """
  x = layers.Conv2DTranspose(n_filters, (2, 2), strides=2, padding="same")(inputs)
  x = layers.Concatenate()([x, skip_features])
  x = conv_block(x, n_filters, dropout, batch_norm)
  return x


#Builds the full u-net model given shape and number of classes.
def build_unet(input_shape, n_classes):
  """
  Build U-Net model architecture
  
  Parameters
    input_shape (tuple): Shape of the input images (height, width, channels)
    n_classes (int): Number of output classes
    
  Returns
    model (tf.keras.Model): Constructed U-Net model
  """
  inputs = layers.Input(shape=input_shape)

  s1, p1 = encoder_block(inputs, 64, dropout=0.0, batch_norm=True)
  s2, p2 = encoder_block(p1, 128, dropout=0.1, batch_norm=True)
  s3, p3 = encoder_block(p2, 256, dropout=0.1, batch_norm=True)
  s4, p4 = encoder_block(p3, 512, dropout=0.1, batch_norm=True)
  b1 = conv_block(p4, 1024, dropout=0.0, batch_norm=True)
  d1 = decoder_block(b1, s4, 512, dropout=0.1, batch_norm=True)
  d2 = decoder_block(d1, s3, 256, dropout=0.1, batch_norm=True)
  d3 = decoder_block(d2, s2, 128, dropout=0.1, batch_norm=True)
  d4 = decoder_block(d3, s1, 64, dropout=0.0, batch_norm=True)

  outputs = layers.Conv2D(n_classes, 3, padding="same", activation="softmax")(d4)
  model = Model(inputs, outputs, name="U-Net")
  return model
