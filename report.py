import pandas as pd
import matplotlib.pyplot as plt
from os import path
from os import makedirs

import gaussian_blur as gbc

dir_path = 'reports'

def create_dir(dir_path):
    if not path.exists(dir_path):
        makedirs(dir_path)

def plot_cpu_gpu_time(name='fig1.png', df_name=['dataframe.csv']):
    df_cpu = pd.read_csv(dir_path + "/" + df_name[0])
    df_gpu = pd.read_csv(dir_path + "/" + df_name[1])

    kernel_size = df_cpu['kernel_size'].unique()

    for ks in kernel_size:
        plt.clf()
        df_cpu_aux = df_cpu.loc[df_cpu['kernel_size']==ks]
        df_gpu_aux = df_gpu.loc[df_gpu['kernel_size']==ks]

        df_cpu_aux = df_cpu_aux.sort_values(by=['sigma_value'])
        df_gpu_aux = df_gpu_aux.sort_values(by=['sigma_value'])

        x = df_cpu_aux['sigma_value']
        cpu_y = df_cpu_aux['time']
        gpu_y = df_gpu_aux['time']

        plt.plot(x, cpu_y, ls='--', marker='o')
        plt.plot(x, gpu_y, ls='--', marker='o')

        plt.xlabel('Valores de Sigma')
        plt.ylabel('Tiempo de ejecución')
        plt.legend(['CPU', 'GPU'])
        plt.title(f'Comparación de tiempos de ejecución CPU vs GPU con kernel de {ks}x{ks}')

        image_name = name+'cpu_vs_gpu_'+str(ks)+'.png'

        create_dir(dir_path)
        plt.savefig(dir_path + '/' + image_name, dpi=300)

def main():
    # plot_cpu_gpu_time(name='rgb_', df_name=['times_cpu_bk.csv', 'times_gpu_bk.csv'])
    # plot_cpu_gpu_time(name='gray_', df_name=['times_cpu.csv', 'times_gpu.csv'])
    image, image_size = gbc.imread('images/astro.jpg')
    gbc.save_image(image)



if __name__ == '__main__':
    main()