#çağırılan kütüphaneler
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
#anaprogramı barındıran def fonksiyonu
def detect_and_predict_mask(frame, faceNet, maskNet):
	# çerçevenin boyutlarını yakalama ve ondan bir damla oluşturma kodu
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# ağ üzerinden yüz algılamalarını alma kodu
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# yüz listemizi, bunlara karşılık gelen konumları ve yüz maskesi ağımızdaki tahminlerin listesini başlatma kodu
	faces = []
	locs = []
	preds = []

	# Yüz algılma kontrol kodu
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
        #yüzün x y kordinatları kodu
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# sınırlayıcı kutuların çerçevenin boyutları dahilinde olmasını sağlayan kod
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# yüzü ve sınırlayıcı kutuları ilgili listelerine ekleme
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# En az 1 yüz bulunduğunda tarama yapma kodu
	if len(faces) > 0:

		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

# serileştirilmiş yüz dedektörü modelimizi diskten yükleme
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# yüz maskesi dedektör modelini diskten yükleme kodu
maskNet = load_model("mask_detector.model")

# kamera açma
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# video akışındaki karelerin üzerinden geçme
while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# tespit edilen yüz konumları ve bunlara karşılık gelen konumlar üzerinde döngü kodu
	for (box, pred) in zip(locs, preds):
		# sınırlayıcı kutuyu ve tahminleri paketinden çıkarma kodu
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Sınırlayıcı kutuyu ve metni çizmek için kullanacağımız sınıf etiketini ve rengini belirleme kodu
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# etiket ve sınırlayıcı kutu dikdörtgenini çıktı karesinde görüntüleme kodu
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# çıktı çerçevesini göster
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# "q" tuşuna basıldıysa döngüden çıkın
	if key == ord("q"):
		break

# program sonu
cv2.destroyAllWindows()
vs.stop()