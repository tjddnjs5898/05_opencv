import cv2 
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar
import webbrowser  # 웹사이트 열기 위한 모듈

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)

    for d in decoded:
        x, y, w, h = d.rect
        barcode_data = d.data.decode('utf-8')
        barcode_type = d.type

        text = '%s (%s)' % (barcode_data, barcode_type)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0 ), 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 'qt'가 들어 있으면 웹브라우저 열기 (한 번만)
        if 'http' in barcode_data.lower():  # 대소문자 구분 없이
            print(f"Opening URL: {barcode_data}")
            webbrowser.open(barcode_data)
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow('camera', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()
