import gaussian_blur as gbc
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-w", "--where", type=str,
                        help="Indicar dónde se ejecutará el algoritmo. cpu | gpu")

    parser.add_argument("-ks", "--kernelSize", type=int,
                        help="Indicar el tamaño del kernel para el filtro Gaussiano.")
    
    parser.add_argument("-sv", "--sigmaValue", type=int,
                        help="Indicar el valor de la desviación estándar")

    parser.add_argument("-in", "--imageName", type=str,
                        help="Indicar el nombre de la imagen final")

    return vars(parser.parse_args())


def main():
    args = get_args()

    if not args['where'] == None:
        kernel_size = args['kernelSize']
        sigma_value = args['sigmaValue']
        name = args['imageName']
        
        if args['where'] == 'cpu':
            gbc.run_with_cpu('images/astro.jpg', kernel_size, sigma_value, name)

        if args['where'] == 'gpu':
            gbc.run_with_cuda('images/astro.jpg', kernel_size, sigma_value, name)

if __name__ == '__main__':
    main()