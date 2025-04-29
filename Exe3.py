import cv2

imagem = cv2.imread('imagem1.jpg', cv2.IMREAD_GRAYSCALE)

sift1 = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
keypoints1, descriptors1 = sift1.detectAndCompute(imagem, None)
imagem_sift1 = cv2.drawKeypoints(imagem, keypoints1, None)

sift2 = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=5)
keypoints2, descriptors2 = sift2.detectAndCompute(imagem, None)
imagem_sift2 = cv2.drawKeypoints(imagem, keypoints2, None)

sift3 = cv2.SIFT_create(contrastThreshold=0.08, edgeThreshold=20)
keypoints3, descriptors3 = sift3.detectAndCompute(imagem, None)
imagem_sift3 = cv2.drawKeypoints(imagem, keypoints3, None)

cv2.imshow('SIFT 1', imagem_sift1)
cv2.imshow('SIFT 2', imagem_sift2)
cv2.imshow('SIFT 3', imagem_sift3)
cv2.waitKey(0)
cv2.destroyAllWindows()
