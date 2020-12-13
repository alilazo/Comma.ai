import cv2
import numpy as np
import os, shutil
from matplotlib import pyplot as plt

print ("OpenCv - " + cv2.__version__)
print ("Numpy - " + np.__version__)
print ("MatPlotLib")

def showFrame(name, frame, grayScale):
	if(grayScale == True):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow(name, gray)
	else:
		cv2.imshow(name, frame)

def __draw_label(img, text, pos):
	font_face = cv2.FONT_HERSHEY_SIMPLEX
	scale = 1
	color = (255,255,255)
	thickness = cv2.FILLED
	margin = 2

	text_size = cv2.getTextSize(text, font_face, scale, thickness)

	end_x = pos[0] + text_size[0][0] + margin
	end_y = pos[1] - text_size[0][1] - margin

	#cv2.rectangle(img, pos, (end_x, end_y), (255,0,0), thickness)
	cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def deleteDirectory(path):
	try:
		shutil.rmtree(path)
		print("Deleted: " + path + "\\ Directory")
	except Exception as e:
		print("Failed to Delete %s. Reason: %s" % (path, e))

def OrbAlgorithm(img):
	orb = cv2.ORB()
	kp = orb.detect(img, None)
	kp, des = orb.compute(img, kp)
	img2 = cv2.drawKeypoints(img, kp, color=(0,255,0), flags=0)
	cv2.imshow("Image Window",img2)

def cornerHarris(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)
	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst.max()] = [0,255,255]
	#plt.imshow(img),plt.show()
	cv2.imshow("Image Window",img)

file = 'train.mp4'
Speed_File = open('train.txt','r')
frame_filePath = 'Frames\\';

if os.path.exists('./Frames'):
	pass
else:
	os.mkdir('Frames')

print("Using: " + file + "\n")
video = cv2.VideoCapture(file)

if (video.isOpened() == False):
	print("Error Opening Video");

#Video Update Every Frame
while(video.isOpened()):
	ret, frame = video.read()
	
	if(ret == True):
		pos_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
		Lines = Speed_File.readline()
		
		__draw_label(frame, str(pos_frame), (300,25))
		__draw_label(frame, "{}".format(Lines.strip()), (220,450))
		
		showFrame('Frame', frame, False)
		cv2.imwrite(frame_filePath + "frame%d.jpg" % pos_frame, frame)
		print("Showing Frame: " + str(pos_frame))
		
		#-- Draws Image --#

		img = cv2.imread(frame_filePath + "frame%d.jpg" % pos_frame)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray,50,150,apertureSize = 3)
		minLineLength = 100
		maxLineGap = 10
		lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
		for x1,y1,x2,y2 in lines[0]:
		    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
		#qOrbAlgorithm(img)
		cornerHarris(img)
		#showFrame("Image Window", img, False)

		cv2.waitKey(500) #--Change to 500 to move automatically
	else:
		break

	#-- Exit Command --#
	if(cv2.waitKey(500) & 0xFF == ord('q')):
		print("Quit On Frame: " + str(pos_frame))

		for filename in os.listdir(frame_filePath):
			print("\t|_" + filename)
		
		cv2.destroyWindow('Image Window')
		deleteDirectory('Frames') #---Deletes Directory---#
		break


	#Ends video when frame == frame count
	if (video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT)):
		print("Finished Video on frame: " + str(pos_frame))
		break

video.release()
Speed_File.close()
cv2.destroyAllWindows()