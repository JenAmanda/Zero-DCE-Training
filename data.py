import tensorflow as tf
import os
class Dataloader:
    def __init__(self, dataPath,
                 max_pixel=255.0,
                 resize_h=512, 
                 resize_w=512, 
                 batch=8, 
                 prefetch=2, 
                 trainsplit=0.8015, 
                 cropsize=128, 
                 standardize=False
                ):

        self.height = resize_h
        self.width = resize_w
        self.max_pixel = max_pixel
        self.batch = batch
        self.prefetch = prefetch
        self.dataPath = dataPath
        self.trainsplit = trainsplit
        self.cropsize = cropsize
        self.standardize = standardize
    
    # Range -1 to 1 with mean 0    
    def _standardize(self, img):
        return ((img * (2.0/self.max_pixel)) - 1.0)
    
    # Range 0 to 1 with mean 0.5
    def _normalize(self, img):
        return img * (1.0/self.max_pixel)
    
    # Load image and perform basic augmentation
    @tf.autograph.experimental.do_not_convert
    def _transform_data(self, img, resize=False, random_crop=False, flips=False):
        input_image = tf.io.read_file(img)
        input_image = tf.cast(tf.io.decode_jpeg(input_image, channels=3), dtype=tf.float32)
        if resize:
            input_image = tf.image.resize(input_image, [self.height, self.width])
        if not resize and random_crop:
            input_image = tf.image.random_crop(input_image, size=[self.cropsize, self.cropsize, input_image.shape[2]])
        if flips:
            tf.cond(tf.random.uniform(()) > 0.5, lambda: tf.image.flip_left_right(input_image), lambda: input_image)
            tf.cond(tf.random.uniform(()) > 0.5, lambda: tf.image.flip_up_down(input_image), lambda: input_image)
        if self.standardize:
            input_image = self._standardize(input_image)
        else:
            input_image = self._normalize(input_image)
        return input_image

    # Generate train and validation data
    def prepareDataset(self, resize=True, random_crop=False, flips=True):
        import glob, os

        # Supported image extensions
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]

        # Gather all image file paths
        all_files = []
        for ext in image_extensions:
            all_files.extend(glob.glob(os.path.join(self.dataPath, ext)))

        # Safety check
        if len(all_files) == 0:
            raise ValueError(f"No image files found in {self.dataPath}")

        # Create TensorFlow dataset from file paths
        dataset_images = tf.data.Dataset.from_tensor_slices(all_files)

        # Shuffle dataset
        buffer_size = len(all_files)
        dataset_images = dataset_images.shuffle(buffer_size=buffer_size, seed=1234, reshuffle_each_iteration=False)

        # Split into training and validation
        train_count = int(buffer_size * self.trainsplit)
        training_images = dataset_images.take(train_count)
        validation_images = dataset_images.skip(train_count)

        # Shuffle training images again for each epoch
        training_images = training_images.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        # Map image transformations
        training_images = training_images.map(
            lambda inp: self._transform_data(inp, resize=resize, random_crop=random_crop, flips=flips),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.batch).prefetch(self.prefetch)

        validation_images = validation_images.map(
            lambda inp: self._transform_data(inp, resize=resize, random_crop=random_crop, flips=flips),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.batch).prefetch(self.prefetch)

        return training_images, validation_images


    
    # Generate Test data
    def prepareTestDataset(self):
        dataset_images = tf.data.Dataset.list_files(os.path.join(self.dataPath, "*.*"))

        testing_images = dataset_images.map(lambda inp: self._transform_data(inp), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testing_images = testing_images.batch(1)
        testing_images = testing_images.prefetch(self.prefetch)
        return testing_images