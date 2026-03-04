import numpy as np
import random
import cv2
from tensorflow.keras.utils import Sequence

# os.makedirs('./data/raw_images', exist_ok=True)
# !wget -c http://images.cocodataset.org/zips/val2017.zip
# !unzip -q -j val2017.zip -d ./data/raw_images/

class HomographyDataset(Sequence):
  def __init__(self, image_list, batch_size = 32, patch_size=32, steps_per_epoch=1000, **kwargs):
    super().__init__(**kwargs)
    self.image_list = image_list
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.steps_per_epoch = steps_per_epoch

    if len(self.image_list) == 0:
      raise ValueError(f"{image_list} folder has no images")

    print(f"Initialized {len(self.image_list)} images")


  def __len__(self):
    return self.steps_per_epoch


  def augment(self, patch):
    patch = patch.astype(np.int16)

    if random.random() < 0.5:
      delta_brightness = random.randint(-30, 30)
      patch = np.clip(patch + delta_brightness, 0, 255)

    if random.random() < 0.5:
      noise = np.random.normal(0, 5, patch.shape)
      patch = np.clip(patch + noise, 0, 255)

    return patch.astype(np.uint8)


  def __getitem__(self, index):
    anchors = []
    positives = []
    negatives = []

    count = 0

    while count < self.batch_size:
      rnd_img_path = random.choice(self.image_list)
      rnd_img = cv2.imread(rnd_img_path, cv2.IMREAD_GRAYSCALE)

      if rnd_img is None:
        continue

      h, w = rnd_img.shape

      if h < 2 * self.patch_size or w < 2 * self.patch_size:
        continue

      pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])

      shift_x = w * 0.2
      shift_y = h * 0.2

      pts2 = pts1 + np.random.uniform(-1, 1, (4, 2)) * [shift_x, shift_y]
      pts2 = pts2.astype(np.float32)

      M = cv2.getPerspectiveTransform(pts1, pts2)

      shifted_img = cv2.warpPerspective(rnd_img, M, (w, h))

      try:
        x_anchor = random.randint(self.patch_size, w - self.patch_size)
        y_anchor = random.randint(self.patch_size, h - self.patch_size)
      except:
        continue

      homo_pnt = np.dot(M, [x_anchor, y_anchor, 1])

      if homo_pnt[2] == 0:
        continue

      homo_pnt[0] /= homo_pnt[2]
      homo_pnt[1] /= homo_pnt[2]
      homo_pnt = homo_pnt.astype(np.uint)

      if homo_pnt[0] < self.patch_size or homo_pnt[0] >= w - self.patch_size:
        continue

      if homo_pnt[1] < self.patch_size or homo_pnt[1] >= h - self.patch_size:
        continue

      while True:
        x_negative = random.randint(self.patch_size, w - self.patch_size)
        y_negative = random.randint(self.patch_size, h - self.patch_size)

        dist = np.sqrt((x_anchor - x_negative) ** 2 + (y_anchor - y_negative) ** 2)

        if dist > self.patch_size * 1.5:
          break

      d = self.patch_size // 2

      anchor_patch = rnd_img[y_anchor - d : y_anchor + d, x_anchor - d : x_anchor + d]
      positive_patch = shifted_img[homo_pnt[1] - d : homo_pnt[1] + d, homo_pnt[0] - d : homo_pnt[0] + d]
      negative_patch = rnd_img[y_negative - d : y_negative + d, x_negative - d : x_negative + d]

      if anchor_patch.shape != (32, 32) or positive_patch.shape != (32, 32) or negative_patch.shape != (32, 32):
        continue

      anchor_patch = self.augment(anchor_patch)
      positive_patch = self.augment(positive_patch)
      negative_patch = self.augment(negative_patch)

      anchors.append(anchor_patch)
      positives.append(positive_patch)
      negatives.append(negative_patch)

      count += 1

    anchors = np.array(anchors).astype('float32') / 255.0
    positives = np.array(positives).astype('float32') / 255.0
    negatives = np.array(negatives).astype('float32') / 255.0

    anchors = np.expand_dims(anchors, axis=-1)
    positives = np.expand_dims(positives, axis=-1)
    negatives = np.expand_dims(negatives, axis=-1)

    return (anchors, positives, negatives), np.zeros((self.batch_size))