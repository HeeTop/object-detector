import cv2
import argparse



ap = argparse.ArgumentParser()
ap.add_argument('-s','--save', required=False,
                help = 'path to save file')


args = ap.parse_args()


# video start

cap = cv2.VideoCapture(0)
# Set parameters

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# We convert the resolutions from float to integer.

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
if args.save:
    out = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
else:
    out = cv2.VideoWriter('data/demo2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


print("종료 : q")
while(True):

    ret, white_image = cap.read()

    # image save
    out.write(white_image)

    # Output Part
    cv2.imshow("Original",white_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()

cv2.destroyAllWindows()
