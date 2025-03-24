import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from skimage.segmentation import active_contour, clear_border, flood_fill
from skimage.filters import (
    threshold_otsu,
    sobel,
    roberts,
    apply_hysteresis_threshold,
    butterworth,
)

# %%
working_dir = r"A:\Marvin\rpi_captures"

image_names = [
    "square_410nm",
    "square_460nm",
    "square_535nm",
    "square_560nm",
    "square_610nm",
]
images = []
for image_name in image_names:
    images.append(ski.io.imread(working_dir + "\\" + image_name + ".jpg"))


# %% plot image histograms
fig, ax = plt.subplots(1, 5, figsize=(20, 5))
for i, image in enumerate(images):
    ax[i].hist(image.flatten(), bins=255)
    ax[i].set_title(image_names[i])

fig.show()

# %%
threshold = 0.5
binary_images = []
zscore_images = []
channel_ids = [2, 2, 1, 1, 0]
images_correct = []
for idx, image in enumerate(images):
    images_correct.append(image[:, :, channel_ids[idx]])
    # binary_images.append(zscore_image > threshold_otsu(zscore_image))
    # binary_images.append(np.max(zscore_image > threshold_otsu(image), axis=2))

# %% plot binary images
fig, ax = plt.subplots(1, 5, figsize=(20, 5))
for i, image in enumerate(binary_images):
    ax[i].imshow(image.astype(int), cmap="seismic", vmin=0, vmax=1)
    ax[i].set_title(image_names[i])
fig.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
ax.imshow(
    binary_images[1].astype(int) - binary_images[3].astype(int),
    cmap="seismic",
    vmin=0,
    vmax=1,
)
fig.show()
# %%
segmentations = []
for idx, image in enumerate(images):
    segmentations.append(sobel(image))

norm_segmentations = []
for segmentation in segmentations:
    norm_segmentations.append(segmentation / np.max(segmentation))
# %%
fig, ax = plt.subplots(1, 5, figsize=(20, 5))
for i, image in enumerate(segmentations):
    ax[i].imshow(np.sum(image, axis=2), cmap="gray")
    ax[i].set_title(image_names[i])


fig.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
cax = ax.imshow(
    np.sum(norm_segmentations[1], axis=2) - np.sum(norm_segmentations[2], axis=2),
    cmap="seismic",
)
fig.colorbar(cax)
fig.show()

# %%
fig, ax = plt.subplots(2, 5, figsize=(30, 10), sharex=True, sharey="row")
for i, image in enumerate(images):
    ax[0, i].plot(np.mean(image[1000:1200, 1500:2300, channel_ids[i]], axis=0))
    ax[0, i].set_title(image_names[i])
    ax[1, i].imshow(image[1000:1200, 1500:2300, channel_ids[i]], cmap="gray")

fig.show()

# %% frequency analysis
magnitudes = []
fft_transforms = []
phase = []
for idx, image in enumerate(images):
    fourier_t = np.fft.fft2(image[1000:1200, 1500:2300, channel_ids[idx]])
    f_shift = np.fft.fftshift(fourier_t)
    fft_transforms.append(f_shift)
    # Compute the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    magnitudes.append(magnitude_spectrum)
    # Compute the phase spectrum
    phase_spectrum = np.angle(f_shift)
    phase.append(phase_spectrum)

# %%
fig, ax = plt.subplots(3, 5, figsize=(20, 5))
for i, image in enumerate(magnitudes):
    ax[0, i].imshow(image, cmap="gray")
    ax[0, i].set_title(image_names[i])
    ax[1, i].imshow(phase[i], cmap="gray")
    ax[1, i].set_title(image_names[i])
    ax[2, i].plot(np.sum(phase[i], axis=0))

fig.show()
# %% Plot fourier transforms
b_filter = []
for idx, image in enumerate(images):
    b_filter.append(
        butterworth(
            image, cutoff_frequency_ratio=0.01, high_pass=True, order=1, channel_axis=2
        )
    )
# %%
fig, ax = plt.subplots(2, 5, figsize=(20, 5))
for i, image in enumerate(b_filter):
    ax[0, i].imshow(image, cmap="gray")
    ax[0, i].set_title(image_names[i])
    ax[1, i].plot(np.sum(image, axis=(2, 0)))
fig.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(np.sum(b_filter[1][:, 1600:2100], axis=(2, 0)), label="460")
ax.plot(np.sum(b_filter[3][:, 1600:2100], axis=(2, 0)), label="560")
fig.show()
