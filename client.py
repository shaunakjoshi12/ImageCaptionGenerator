import socket

socket = socket.socket()

socket.connect(("localhost", 8888))

image = raw_input("Enter Image Name: ")
socket.send(image)

predictedCaptions = socket.recv(1024)

print "Predicted Captions: ", predictedCaptions.split("\t")[0]
print "Time Taken: ", predictedCaptions.split("\t")[1]

socket.close()
