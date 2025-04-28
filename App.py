import cv2
import numpy as np

def detectar_cantos_harris(imagem_path, block_size, ksize, k, limiar, janela_nome, salvar_path):
    imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        return
    harris = cv2.cornerHarris(imagem, block_size, ksize, k)
    harris = cv2.dilate(harris, None)
    imagem_com_cantos = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    imagem_com_cantos[harris > limiar * harris.max()] = [0, 0, 255]
    cv2.imshow(janela_nome, imagem_com_cantos)
    cv2.imwrite(salvar_path, imagem_com_cantos)

imagens = ['imagem1.jpg', 'imagem2.jpg', 'imagem3.jpg']

parametros = [
    {'block_size': 2, 'ksize': 3, 'k': 0.04},
    {'block_size': 8, 'ksize': 1, 'k': 0.05},
    {'block_size': 9, 'ksize': 5, 'k': 0.06}
]

limiar_fixo = 0.01

for idx_img, imagem_path in enumerate(imagens, start=1):
    for idx_param, param in enumerate(parametros, start=1):
        janela_nome = f"Imagem {idx_img} - Config {idx_param}"
        salvar_path = f"saida_imagem{idx_img}_config{idx_param}.jpg"
        detectar_cantos_harris(
            imagem_path,
            block_size=param['block_size'],
            ksize=param['ksize'],
            k=param['k'],
            limiar=limiar_fixo,
            janela_nome=janela_nome,
            salvar_path=salvar_path
        )

cv2.waitKey(0)
cv2.destroyAllWindows()
