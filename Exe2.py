import cv2
import numpy as np

def detectar_cantos_shi_tomasi(imagem_path, max_corners, quality_level, min_distance, janela_nome, salvar_path):
    imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        return
    cantos = cv2.goodFeaturesToTrack(imagem, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    cantos = np.int0(cantos)
    imagem_com_cantos = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    for canto in cantos:
        x, y = canto.ravel()
        cv2.circle(imagem_com_cantos, (x, y), 3, (0, 0, 255), -1)
    cv2.imshow(janela_nome, imagem_com_cantos)
    cv2.imwrite(salvar_path, imagem_com_cantos)

imagens = ['imagem1.jpg', 'imagem2.jpg', 'imagem3.jpg']

parametros = [
    {'max_corners': 30, 'quality_level': 0.01, 'min_distance': 20},
    {'max_corners': 100, 'quality_level': 0.05, 'min_distance': 10},
    {'max_corners': 300, 'quality_level': 0.001, 'min_distance': 2}
]

for idx_img, imagem_path in enumerate(imagens, start=1):
    for idx_param, param in enumerate(parametros, start=1):
        janela_nome = f"Imagem {idx_img} - Config {idx_param}"
        salvar_path = f"saida_shitomasi_imagem{idx_img}_config{idx_param}.jpg"
        detectar_cantos_shi_tomasi(
            imagem_path,
            max_corners=param['max_corners'],
            quality_level=param['quality_level'],
            min_distance=param['min_distance'],
            janela_nome=janela_nome,
            salvar_path=salvar_path
        )

cv2.waitKey(0)
cv2.destroyAllWindows()
