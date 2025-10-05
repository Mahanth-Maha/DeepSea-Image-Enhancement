import os
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt

### constants
K = 2**8  
SHOW_IMAGES = True
FONT_SIZE = 16
FIG_WIDTH = 12

#### image Processing
def load_image(filepath):
    img = plt.imread(filepath)
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img

def to_grayscale(img):
    if img.ndim == 2:
        return img
    r, g, b = img[:,:, 0], img[:,:, 1], img[:,:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.float64)


#### PLOT-ing

def plot_image(image, title = '', save = None):
    plt.figure(figsize=(FIG_WIDTH//2,3* (FIG_WIDTH//4)))
    plt.imshow(image, cmap='gray')
    plt.title(title, fontsize=FONT_SIZE)
    plt.axis('off')
    if save:
        plt.savefig(save, bbox_inches='tight')
    if SHOW_IMAGES:
        plt.show() 
    plt.tight_layout()
    plt.close()

def plot_img_2_pair(img1, img2, title1 = '', title2 = '', save = None, cmap = None):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH // 2))
    axes[0].imshow(img1, cmap=cmap if img1.ndim == 2 else None)
    axes[0].set_title(title1, fontsize=FONT_SIZE)
    axes[0].axis('off')
    axes[1].imshow(img2, cmap=cmap if img2.ndim == 2 else None)
    axes[1].set_title(title2, fontsize=FONT_SIZE)
    axes[1].axis('off')
    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(wspace=0.05)
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=150)
    if SHOW_IMAGES:
        plt.show()

    plt.close(fig)

def plot_img_3_pair(img1, img2, img3, title1='', title2='', title3='', save=None, cmap='gray'):
    images = [img1, img2, img3]
    titles = [title1, title2, title3]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_WIDTH // 3))

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
        ax.set_title(title, fontsize=int(FONT_SIZE * 0.9))
        ax.axis('off')

    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(wspace=0.05)

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=150)
    if SHOW_IMAGES:
        plt.show()
    plt.close(fig)

def plot_img_4_pair(img1, img2, img3, img4, title1='', title2='', title3='', title4='', save=None, cmap='gray'):
    images = [img1, img2, img3, img4]
    titles = [title1, title2, title3, title4]

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, int(FIG_WIDTH * 0.75)))
    axes = axes.flatten()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
        ax.set_title(title, fontsize=int(FONT_SIZE * 0.9))
        ax.axis('off')

    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=150)
    if SHOW_IMAGES:
        plt.show()
    plt.close(fig)


def plot_hist(img_hist, title='Histogram', save=None, otsu_thresh = None):
    plt.figure(figsize =(3*FIG_WIDTH//4,2*FIG_WIDTH//4))
    if isinstance(img_hist, np.ndarray):
        img_hist = img_hist.ravel()
    plt.bar(range(len(img_hist)), img_hist)
    plt.title(title, fontsize=FONT_SIZE)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0 - 5, len(img_hist) - 1 + 5])
    plt.xticks(np.arange(0, len(img_hist)+1, step=len(img_hist)//8))
    if otsu_thresh :
        plt.axvline(x=otsu_thresh, color='r', linestyle='--', label=f'Otsu Threshold = {otsu_thresh}')
        plt.legend(fontsize=FONT_SIZE)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save)
    if SHOW_IMAGES:
        plt.show()
    plt.close()

def plot_hist_2_pair(img_hist1, img_hist2, title1='Histogram 1', title2='Histogram 2', save=None, otsu_thresh1=None, otsu_thresh2=None):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH//2))
    if isinstance(img_hist1, np.ndarray):
        img_hist1 = img_hist1.ravel()
    if isinstance(img_hist2, np.ndarray):
        img_hist2 = img_hist2.ravel()
    axes[0].bar(range(len(img_hist1)), img_hist1)
    axes[0].set_title(title1, fontsize=FONT_SIZE)
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim([-5, len(img_hist1)-1+5])
    axes[0].set_xticks(np.arange(0, len(img_hist1)+1, step=max(1, len(img_hist1)//8)))
    if otsu_thresh1 is not None:
        axes[0].axvline(x=otsu_thresh1, color='r', linestyle='--', label=f'Otsu Threshold = {otsu_thresh1}')
        axes[0].legend(fontsize=FONT_SIZE)
    axes[0].grid()
    axes[1].bar(range(len(img_hist2)), img_hist2)
    axes[1].set_title(title2, fontsize=FONT_SIZE)
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim([-5, len(img_hist2)-1+5])
    axes[1].set_xticks(np.arange(0, len(img_hist2)+1, step=max(1, len(img_hist2)//8)))
    if otsu_thresh2 is not None:
        axes[1].axvline(x=otsu_thresh2, color='r', linestyle='--', label=f'Otsu Threshold = {otsu_thresh2}')
        axes[1].legend(fontsize=FONT_SIZE)
    axes[1].grid()
    plt.tight_layout()
    if save:
        plt.savefig(save)
    if SHOW_IMAGES:
        plt.show()
    plt.close(fig)
    
def plot_hist_curve(img_hist, title='Histogram (Curve)', save=None, otsu_thresh=None):
    plt.figure(figsize=(3*FIG_WIDTH//4, 2*FIG_WIDTH//4))
    
    if isinstance(img_hist, np.ndarray):
        img_hist = img_hist.ravel()
    x = np.arange(len(img_hist))

    plt.plot(x, img_hist, color='blue', linewidth=2)
    plt.title(title, fontsize=FONT_SIZE)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([-5, len(img_hist)-1+5])
    plt.xticks(np.arange(0, len(img_hist)+1, step=max(1, len(img_hist)//8)))

    if otsu_thresh is not None:
        plt.axvline(x=otsu_thresh, color='r', linestyle='--', label=f'Otsu Threshold = {otsu_thresh}')
        plt.legend(fontsize=FONT_SIZE)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    if SHOW_IMAGES:
        plt.show()
    plt.close()


def plot_hist_curve_2_pair(img_hist1, img_hist2, title1='Histogram 1 (Curve)', title2='Histogram 2 (Curve)', overlay_1_on_2 = False, save=None, otsu_thresh1=None, otsu_thresh2=None):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH//2))
    if isinstance(img_hist1, np.ndarray):
        img_hist1 = img_hist1.ravel()
    if isinstance(img_hist2, np.ndarray):
        img_hist2 = img_hist2.ravel()
    x1 = np.arange(len(img_hist1))
    x2 = np.arange(len(img_hist2))
    axes[0].plot(x1, img_hist1, color='blue', linewidth=2)
    axes[0].set_title(title1, fontsize=FONT_SIZE)
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim([-5, len(img_hist1)-1+5])
    axes[0].set_xticks(np.arange(0, len(img_hist1)+1, step=max(1, len(img_hist1)//8)))
    if otsu_thresh1 is not None:
        axes[0].axvline(x=otsu_thresh1, color='r', linestyle='--', label=f'Otsu Threshold = {otsu_thresh1}')
        axes[0].legend(fontsize=FONT_SIZE)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].plot(x2, img_hist2, color='green', linewidth=2)
    if overlay_1_on_2:
        axes[1].plot(x1, img_hist1, color='blue', linewidth=0.25)
    axes[1].set_title(title2, fontsize=FONT_SIZE)
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim([-5, len(img_hist2)-1+5])
    axes[1].set_xticks(np.arange(0, len(img_hist2)+1, step=max(1, len(img_hist2)//8)))
    if otsu_thresh2 is not None:
        axes[1].axvline(x=otsu_thresh2, color='r', linestyle='--', label=f'Otsu Threshold = {otsu_thresh2}')
        axes[1].legend(fontsize=FONT_SIZE)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save:
        plt.savefig(save)
    if SHOW_IMAGES:
        plt.show()
    plt.close(fig)


def plot_kernel(kernel, type_of='',save=None,):
    plt.figure(figsize=(FIG_WIDTH//2,FIG_WIDTH//2))
    min_, max_ = kernel.min(), kernel.max()
    plt.imshow(kernel, cmap='RdYlGn', vmin=-max(abs(min_), abs(max_)), vmax=max(abs(min_), abs(max_)))
    plt.colorbar()
    plt.title(f'{type_of}Kernel \n(Positives Green, Negatives Red)')
    plt.tight_layout()
    if save:
        plt.savefig(save)
    if SHOW_IMAGES:
        plt.show()
    plt.close()

def plot_curve(x,y, title='', xlabel='', ylabel='', save=None):
    plt.figure(figsize=(3*FIG_WIDTH//4,2*FIG_WIDTH//4))
    plt.plot(x,y)
    plt.title(title, fontsize=FONT_SIZE)
    plt.xlabel(xlabel, fontsize=int(FONT_SIZE * 0.8))
    plt.ylabel(ylabel, fontsize=int(FONT_SIZE * 0.8))
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save)
    if SHOW_IMAGES:
        plt.show()
    plt.close()

def plot_diff(diff, title='Difference', save=None):
    plt.figure(figsize=(FIG_WIDTH//2,FIG_WIDTH//2))
    _ = plt.imshow(diff, vmin=0, vmax=255, cmap='gray')
    _ = plt.title(title, fontsize=FONT_SIZE)
    _ = plt.colorbar()
    plt.tight_layout()
    if save:
        plt.savefig(save)
    if SHOW_IMAGES:
        plt.show()
    plt.close()



#### KERNELs

def identity_kernel(size=3):
    k = np.zeros((size, size))
    k[size//2, size//2] = 1
    return k , 'identity'

def box_blur_kernel(size=3):
    return np.ones((size, size)) / (size * size) , 'box_blur'

def gaussian_kernel(size=5, sigma=1):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel) , 'gaussian'

def sobel_kernel_x():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) , 'sobel_x'

def sobel_kernel_y():
    return np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]]) , 'sobel_y'

def laplacian_kernel():
    return np.array([[0, -1,  0],[-1, 4, -1],[0, -1,  0]]) , 'laplacian'

def sharpen_kernel():
    return np.array([[0, -1,  0],[-1, 5, -1],[0, -1,  0]]) , 'sharpen'

def emboss_kernel():
    return np.array([[-2, -1, 0],[-1,  1, 1],[ 0,  1, 2]]) , 'emboss'
