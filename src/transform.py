import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
  means = (0.4914, 0.4822, 0.4465)
  stds = (0.2470, 0.2435, 0.2616)
  return A.Compose([
        A.Normalize(means, stds),
        A.PadIfNeeded(40,40),
        A.RandomCrop(height=32, width=32),
        # A.HorizontalFlip(),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=means, mask_fill_value =None),
        ToTensorV2()])

def get_test_transforms():
  means = (0.4914, 0.4822, 0.4465)
  stds = (0.2470, 0.2435, 0.2616)
  return A.Compose([
        A.Normalize(means, stds), #normalize
        ToTensorV2()])