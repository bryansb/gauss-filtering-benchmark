import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import time
from PIL import Image

import gaussian_blur_source as gbs

MAX_BLOCK_SIZE = 32

def imread(path):
    # image = np.array(Image.open(path), dtype=np.uint8)
    image = np.array(Image.open(path).convert('L'))
    return (image, image.shape)

def save_image(image, name='gauss_image.png'):
    # image = Image.fromarray(image)
    image = Image.fromarray(np.uint8(image), 'L')
    image.save(name)

def split_channels(img):
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]
    return (r_channel.astype(np.int32), g_channel.astype(np.int32), b_channel.astype(np.int32))

def merge_channels(r_channel, g_channel, b_channel, img_size=(1920, 1080)):
    img = np.array( [ [ np.zeros(3, dtype=np.uint8) ] * img_size[1] ] * img_size[0] )
    img[:, :, 0] = r_channel[:]
    img[:, :, 1] = g_channel[:]
    img[:, :, 2] = b_channel[:]
    return img

def apply_gauss_cuda(channel, image_size, kernel):
    start_time = time.time()
    
    # Reserva y copia los datos en memoria
    channel_gpu = cuda.mem_alloc(channel.nbytes)
    cuda.memcpy_htod(channel_gpu, channel)

    # Reserva y copia el kernel gausiano
    kernel_gpu = cuda.mem_alloc(kernel.nbytes)
    cuda.memcpy_htod(kernel_gpu, kernel)

    # Reserva y copia la imagen filtrada
    channel_out = np.empty_like(channel)
    channel_out_gpu = cuda.mem_alloc(channel_out.nbytes)

    # Carga la función correspondiente
    mod, name = gbs.get_gaussian_global_func(image_size[0], image_size[1], kernel.shape[0])
    func = mod.get_function(name)

    # Ejecuta las funciones en Cuda
    bdim = (MAX_BLOCK_SIZE, MAX_BLOCK_SIZE, 1)
    gdim = (int(image_size[1]/MAX_BLOCK_SIZE), int(image_size[0]/MAX_BLOCK_SIZE))
    
    
    func(channel_gpu, kernel_gpu, channel_out_gpu, block=bdim, grid=gdim)

    # Copia desde el device la imagen filtrada
    cuda.memcpy_dtoh(channel_out, channel_out_gpu)

    end_time = time.time()
    final_time = end_time - start_time

    channel_gpu.free()
    kernel_gpu.free()
    channel_out_gpu.free()

    return channel_out, final_time

def run_with_cuda(path, kernel_size, sigma_value, name='gauss_image_cuda.png'):
    # lee una imagen y las separa en canales
    image, image_size = imread(path)
    image = image.astype(np.int32)
    
    #print('IMG SHAPE:', image.shape)
    #image_r, image_g, image_b = split_channels(image)

    kernel = gbs.get_gaussian_kernel(kernlen=kernel_size, nsig=sigma_value)
    #print(kernel)
    
    image_out, final_time = apply_gauss_cuda(image, image_size, kernel)
    #print(image_out)
    #image_r_out, time_r = apply_gauss_cuda(image_r, image_size, kernel)
    #image_g_out, time_g = apply_gauss_cuda(image_g, image_size, kernel)
    #image_b_out, time_b = apply_gauss_cuda(image_b, image_size, kernel)

    # final_time = time_r + time_g + time_b
    
    # image = merge_channels(image_r_out, image_g_out, image_b_out, image_size)
    #pathName = 'images_results/'+name
    pathName = 'images_results_gray/'+name
    save_image(image_out, name=pathName)
    print(f'{image_size[0]},{image_size[1]},{final_time}')
    return image

def get_gauss_value(channel, row, col, kernel):
    k_len = int(kernel.shape[0]/2)
    start_row = row - k_len
    end_row = row + k_len

    start_col = col - k_len
    end_col = col + k_len

    subimage = None
    if start_row >= 0 and end_row < channel.shape[0] \
        and start_col >= 0 and end_col < channel.shape[1]:
        subimage = channel[start_row:end_row + 1, start_col:end_col + 1]

        subimage = subimage * kernel
    return np.sum(subimage)

def apply_gauss_cpu(channel, image_size, kernel):
    k_size = kernel.shape[0]
    channel_out = np.empty_like(channel)

    start_time = time.time()
    for row in range(channel.shape[0]):
        for col in range(channel.shape[1]):
            value = get_gauss_value(channel, row, col, kernel)
            if value != None:
                channel_out[row, col] = value
            else:
                channel_out[row, col] = channel[row, col]
    end_time = time.time()

    final_time = end_time - start_time

    return channel_out, final_time

def run_with_cpu(path, kernel_size, sigma_value, name='gauss_image_cpu.png'):
    image, image_size = imread(path)
    image = image.astype(np.int32)
    #print('IMG SHAPE:', image.shape)
    #image_r, image_g, image_b = split_channels(image)

    kernel = gbs.get_gaussian_kernel(kernlen=kernel_size, nsig=sigma_value)

    image_out, final_time = apply_gauss_cpu(image, image_size, kernel)
    #image_r, time_r = apply_gauss_cpu(image_r, image_size, kernel)
    #image_g, time_g = apply_gauss_cpu(image_g, image_size, kernel)
    #image_b, time_b = apply_gauss_cpu(image_b, image_size, kernel)

    #final_time = time_r + time_g + time_b

    #image = merge_channels(image_r, image_g, image_b, image_size)
    #pathName = 'images_results/'+name
    pathName = 'images_results_gray/'+name
    save_image(image_out, name=pathName)
    print(f'{image_size[0]},{image_size[1]},{final_time}')
    return image