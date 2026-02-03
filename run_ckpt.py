import os
import tensorflow as tf
import numpy as np
from PIL import Image
from absl import flags, app
from model import DCENet  # your trained model class

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", None, "Path to input image or folder of images")
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint")
flags.DEFINE_string("image_savepath", "./outputs", "Where to save enhanced images")
flags.DEFINE_boolean("compute_metrics", False, "Compute PSNR/SSIM against inputs")

def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0  # normalize
    img_np = np.expand_dims(img_np, axis=0)           # shape (1,H,W,3)
    return tf.convert_to_tensor(img_np), img.size     # return size for saving

def main(argv):
    os.makedirs(FLAGS.image_savepath, exist_ok=True)

    # Load model & checkpoint
    model = DCENet()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(FLAGS.checkpoint_path).expect_partial()
    print(f"Loaded checkpoint: {FLAGS.checkpoint_path}")

    # Prepare image files
    if os.path.isdir(FLAGS.data_path):
        files = sorted([os.path.join(FLAGS.data_path, f) for f in os.listdir(FLAGS.data_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    else:
        files = [FLAGS.data_path]

    # Optional metrics
    if FLAGS.compute_metrics:
        running_psnr = tf.metrics.Mean()
        running_ssim = tf.metrics.Mean()

    # Process images
    for i, path in enumerate(files):
        img_tensor, orig_size = load_image(path)
        enhanced, _, _ = model(img_tensor)  

        # Compute metrics
        if FLAGS.compute_metrics:
            psnr = tf.image.psnr(enhanced, img_tensor, max_val=1.0)
            ssim = tf.image.ssim(enhanced, img_tensor, max_val=1.0)
            running_psnr.update_state(psnr)
            running_ssim.update_state(ssim)

        # Save enhanced image at original size
        out_np = (enhanced.numpy()[0] * 255).astype(np.uint8)
        out_img = Image.fromarray(out_np)
        out_img = out_img.resize(orig_size)  # optional, usually already same
        out_name = os.path.join(FLAGS.image_savepath, os.path.basename(path))
        out_img.save(out_name)
        print(f"Saved enhanced image: {out_name}")

    if FLAGS.compute_metrics:
        print(f"Average PSNR: {running_psnr.result():.4f}, Average SSIM: {running_ssim.result():.4f}")

if __name__ == "__main__":
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("checkpoint_path")
    app.run(main)
