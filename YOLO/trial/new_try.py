# Import the two main classes FaceAnalyzer and Face
from FaceAnalyzer import FaceAnalyzer, Face

fa = FaceAnalyzer()
# ... Recover an image in RGB format as numpy array (you can use pillow opencv but if you use opencv make sure you change the color space from BGR to RGB)
# Now process the image
fa.process(image)

# Now you can find faces in fa.faces which is a list of instances of object Face
if fa.nb_faces>0:
    print(f"{fa.nb_faces} Faces found")
    #We can get the face rectangle image like this
    face_image = face.getFaceBox(frame)
    # We can get the face forehead image like this
    forehead_image = face.getFaceBox(frame, face.face_forhead_indices)
