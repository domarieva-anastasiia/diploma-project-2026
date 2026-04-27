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

def split_into_patches(image, patch_size, stride):
    h, w, _ = image.shape
    patches = []
    positions = []

    ys = list(range(0, h - patch_size + 1, stride))
    xs = list(range(0, w - patch_size + 1, stride))

    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)

    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)

    for y in ys:
        for x in xs:
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))

    return patches, positions

def merge_patches(patches, positions, image_shape, patch_size, scale):
    h, w = image_shape
    sr_h = h * scale
    sr_w = w * scale

    result = np.zeros((sr_h, sr_w, 3), dtype=np.float32)
    weight = np.zeros((sr_h, sr_w, 3), dtype=np.float32)

    for patch, (y, x) in zip(patches, positions):
        y0 = y * scale
        x0 = x * scale

        ph, pw, _ = patch.shape

        result[y0:y0+ph, x0:x0+pw] += patch
        weight[y0:y0+ph, x0:x0+pw] += 1.0
    
    weight[weight == 0] = 1.0
    return result / weight

def enhance_large_image(image, model, patch_size=64, overlap=24, scale=4, progress_callback=None):
    stride = patch_size - overlap

    padded, orig_h, orig_w = pad_image(image, patch_size)

    patches, positions = split_into_patches(padded, patch_size, stride)

    sr_patches = []
    total = len(patches)
  

    for patch in patches:
        patch = np.expand_dims(patch, axis=0)
        sr = model(patch, training=False)
        sr = sr.numpy()[0]
        sr_patches.append(sr)

        if progress_callback is not None:
            progress = int((i + 1) / total * 100)
            progress_callback(progress)

    sr_image = merge_patches(
        sr_patches,
        positions,
        padded.shape[:2],
        patch_size,
        scale
    )

    sr_image = sr_image[:orig_h*scale, :orig_w*scale]

    return sr_image