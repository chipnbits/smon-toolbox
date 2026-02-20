import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation


def show_images(tensors, denormalize=True, title="", save_path=None, show_windowed=True):
    """Shows the provided images as sub-pictures in a square"""

    if denormalize:
        images = denormalize_images(tensors)
    else:
        images = tensors

    print(f"images tensor is {images.shape}")
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                # plt.imshow(images[idx][0], cmap="gray")
                # Handle color vs grayscale images
                if images[idx].shape[0] == 1:

                    plt.imshow(images[idx][0], cmap="gray")
                else:
                    # Einops to convert from CxHxW to HxWxC
                    plt.imshow(einops.rearrange(images[idx], "c h w -> h w c"))

                plt.axis("off")
                idx += 1

    fig.suptitle(title, fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300)
    # Showing the figure

    if show_windowed:
        plt.show()


def show_first_batch(loader, denormalize=False):
    for batch in loader:
        if type(batch) is list:
            show_images(batch[0], denormalize)
        else:
            show_images(batch, denormalize)
        break


def denormalize_images(imgs):
    """Normalize all channels of images to [0, 255], returns a clone of the images"""
    imgs = imgs.clone()
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    imgs = (imgs * 255).type(torch.uint8)
    return imgs


def show_time_series(image_frames, save_path=None, denormalize=False):
    """Shows the time series of images as a single image
    Params:
        image_frames: Tensor of shape T x N x C x H x W containing the time series of batch of images
    """
    mosaic = einops.rearrange(image_frames, "t n c h w -> (n h) (t w) c")
    if denormalize:
        mosaic = denormalize_images(mosaic)
    fig = plt.figure(figsize=(16, 8))
    plt.imshow(mosaic.detach().cpu().numpy(), cmap="gray")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def animate_batch(image_frames, save_path=None, denormalize=False, fps=10):
    """Takes a TxNxCxHxW tensor and animates it as a gif with the last frame held for extra frames"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    t, n, c, h, w = image_frames.shape

    if n % 4 != 0:
        raise ValueError("Batch size must be a multiple of 4")

    if save_path and save_path.split(".")[-1] != "gif":
        raise ValueError("Save path must be a .gif file, use the .gif extension")

    def make_mosaic(frame_t):
        b1 = 4
        b2 = frame_t.size(0) // b1
        mosaic = einops.rearrange(frame_t, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=b1, b2=b2)
        if denormalize:
            # Assuming denormalize_images is a function to convert data for display
            mosaic = denormalize_images(mosaic)
        return mosaic.cpu().numpy()

    # Setup initial image
    frame_t = image_frames[0]
    mosaic = make_mosaic(frame_t)
    img_ax = ax.imshow(mosaic, cmap="gray")

    def update(frame_index):
        # Show the last frame for extra frames without recalculating
        if frame_index >= t:
            frame_index = t - 1  # Hold on the last frame

        frame_t = image_frames[frame_index]
        mosaic = make_mosaic(frame_t)
        img_ax.set_data(mosaic)
        ax.set_title(f"Interpolation: Step {frame_index + 1} of {t}")
        return (img_ax,)

    # Extra frames for final image
    end_frames = 40

    # Create animation, adjusting total frames by adding linger frames
    anim = FuncAnimation(fig, update, frames=t + end_frames, interval=1000 / fps, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)  # Close the figure to prevent display issues

    return anim
