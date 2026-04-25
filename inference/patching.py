import numpy as np

def pad_image(image, patch_size):
    h, w, _ = image.shape

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    padded = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='reflect'
    )

    return padded, h, w

def split_into_patches(image, patch_size):
    h, w, _ = image.shape
    patches = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)

    return patches

def merge_patches(patches, image_shape, patch_size, scale):
    h, w = image_shape
    sr_h = h * scale
    sr_w = w * scale

    result = np.zeros((sr_h, sr_w, 3), dtype=np.float32)

    idx = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = patches[idx]
            idx += 1

            y0 = y * scale
            x0 = x * scale

            result[y0:y0+patch.shape[0], x0:x0+patch.shape[1]] = patch

    return result

def enhance_large_image(image, model, patch_size=64, scale=4):
    padded, orig_h, orig_w = pad_image(image, patch_size)

    patches = split_into_patches(padded, patch_size)

    sr_patches = []

    for patch in patches:
        patch = np.expand_dims(patch, axis=0)  # [1,H,W,3]

        sr = model(patch, training=False)
        sr = sr.numpy()[0]

        sr_patches.append(sr)

    sr_image = merge_patches(
        sr_patches,
        padded.shape[:2],
        patch_size,
        scale
    )

    # обрізаємо padding
    sr_image = sr_image[:orig_h*scale, :orig_w*scale]

    return sr_image