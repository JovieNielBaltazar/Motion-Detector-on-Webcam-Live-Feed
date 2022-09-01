import cv2, time, pandas
from datetime import datetime

video = cv2.VideoCapture(0)

first_image = None
status_list = [None, None]
times =[]
df = pandas.DataFrame(columns=["Start", "End"])

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    if first_image is None:
        first_image = gray
        continue

    delta_frame = cv2.absdiff(first_image, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 225, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 3)

    status_list.append(status)

    status_list = status_list[-2:]

    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    cv2.imshow("Gray Image", gray)
    cv2.imshow("Delta Image", delta_frame)
    cv2.imshow("Treshold Image", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        if status == 1:
            times.append(datetime.now())
        break


for x in range(0, len(times), 2):
    df = df.append({"Start": times[x], "End": times[x + 1]}, ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()
