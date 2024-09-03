import cv2

# Charger le classificateur en cascade pour la détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Charger la vidéo
cap = cv2.VideoCapture('VID_20240903_083709290.mp4')

# Obtenir les dimensions de la vidéo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Définir le codec et créer l'objet VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video_floutee.avi', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir chaque frame en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Flouter chaque visage détecté
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (99, 99), 30)
        frame[y:y+h, x:x+w] = face

    # Écrire la frame floutée dans la nouvelle vidéo
    out.write(frame)

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()
