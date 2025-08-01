# OpenCV로 QR 코드 인식하기
- matplot으로 QR 보여지기
<img width="3000" height="3000" alt="image" src="https://github.com/user-attachments/assets/72e2f4ee-5ea7-4000-97e8-b24b47014797" />

~~~
import cv2
import matplotlib.pyplot as plt
#import pyzbar.pyzbar as pyzbar

img = cv2.imread('../img/frame.png')
plt.imshow(img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
~~~
- 디코딩
<img width="642" height="548" alt="image" src="https://github.com/user-attachments/assets/bb5dbf7c-64b9-4b6f-b63b-8a0e654b49db" />

~~~
import cv2
import matplotlib.pyplot as plt
#import pyzbar.pyzbar as pyzbar
import pyzbar.pyzbar as pyzbar

img = cv2.imread('../img/frame.png')
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#plt.imshow(img)
plt.imshow(gray, cmap='gray')
plt.show()

# 디코딩

decoded = pyzbar.decode(gray)
print(decoded)

cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

- 카메라 QR 인식
<img width="642" height="512" alt="image" src="https://github.com/user-attachments/assets/9a7bdb16-50d0-4bfb-b275-5cbc2e526766" />

~~~
import cv2
import cv2 
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar

img = cv2.imread('../img/frame.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.imread('../img/frame.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret ,img = cap.read()

    if not ret:
        continue

    #img = cv2.imread('../img/frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)

    for d in decoded:
        x, y, w, h = d.rect 

        barcode_data = d.data.decode('utf-8')
        barcode_type = d.type
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        #print(d.data.decode('utf-8'))
        #barcode_data = d.data.decode('utf-8')
        #print(d.type)
        #barcode_type = d.type

        text = '%s (%s)' % (barcode_data, barcode_type)

        #cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0 ), 20)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0 ), 2)
        #cv2.putText(img, text, (d.rect[0], d.rect[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )
    
    cv2.imshow('camera', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()
#plt.imshow(img)
plt.imshow(gray, cmap='gray')
plt.show()
#plt.imshow(gray, cmap='gray')
#plt.show()

# 디코딩 
# decoded = pyzbar.decode(gray)
# print(decoded)

# for d in decoded:
#     print(d.data.decode('utf-8'))
#     barcode_data = d.data.decode('utf-8')
#     print(d.type)
#     barcode_type = d.type

#     text = '%s (%s)' % (barcode_data, barcode_type)

# 디코딩
#     cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0 ), 20)
#     cv2.putText(img, text, (d.rect[0], d.rect[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )

decoded = pyzbar.decode(gray)
print(decoded)
# plt.imshow(img)
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
img: 사각형을 그릴 이미지입니다.
pt1: 사각형의 왼쪽 상단 꼭지점 좌표입니다. (x, y) 형식의 튜플이어야 합니다.
pt2: 사각형의 오른쪽 하단 꼭지점 좌표입니다. (x, y) 형식의 튜플이어야 합니다.
color: 사각형의 색상입니다. (B, G, R) 형식의 튜플이나 스칼라 값으로 지정할 수 있습니다.
thickness: 선택적으로 사각형의 선 두께를 지정합니다. 기본값은 1입니다. 음수 값을 전달하면 내부를 채웁니다.
lineType: 선택적으로 선의 형태를 지정합니다. 기본값은 cv2.LINE_8입니다.
shift: 선택적으로 좌표값의 소수 부분을 비트 시프트할 양을 지정합니다.
"""
# cv2.waitKey(0)
# cv2.destroyAllWindows()
~~~

- 글자 크기 줄임
~~~
#cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0 ), 20)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0 ), 2)
#cv2.putText(img, text, (d.rect[0], d.rect[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA )

cv2.imshow('camera', img)
~~~

- QR 인식후 웹 사이트이동
~~~
import cv2 
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar
import webbrowser  # 웹사이트 열기 위한 모듈

#img = cv2.imread('../img/frame.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret ,img = cap.read()

while cap.isOpened():
    ret, img = cap.read()
if not ret:
continue

    #img = cv2.imread('../img/frame.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
decoded = pyzbar.decode(gray)

for d in decoded:
        x, y, w, h = d.rect 

        x, y, w, h = d.rect
barcode_data = d.data.decode('utf-8')
barcode_type = d.type
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        #print(d.data.decode('utf-8'))
        #barcode_data = d.data.decode('utf-8')
        #print(d.type)
        #barcode_type = d.type

text = '%s (%s)' % (barcode_data, barcode_type)

        #cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0 ), 20)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0 ), 2)
        #cv2.putText(img, text, (d.rect[0], d.rect[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA )
    
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
@@ -43,36 +38,3 @@

cap.release() 
cv2.destroyAllWindows()
#plt.imshow(img)
#plt.imshow(gray, cmap='gray')
#plt.show()

# 디코딩 
# decoded = pyzbar.decode(gray)
# print(decoded)

# for d in decoded:
#     print(d.data.decode('utf-8'))
#     barcode_data = d.data.decode('utf-8')
#     print(d.type)
#     barcode_type = d.type

#     text = '%s (%s)' % (barcode_data, barcode_type)

#     cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0 ), 20)
#     cv2.putText(img, text, (d.rect[0], d.rect[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )

# plt.imshow(img)
# plt.show()

"""
img: 사각형을 그릴 이미지입니다.
pt1: 사각형의 왼쪽 상단 꼭지점 좌표입니다. (x, y) 형식의 튜플이어야 합니다.
pt2: 사각형의 오른쪽 하단 꼭지점 좌표입니다. (x, y) 형식의 튜플이어야 합니다.
color: 사각형의 색상입니다. (B, G, R) 형식의 튜플이나 스칼라 값으로 지정할 수 있습니다.
thickness: 선택적으로 사각형의 선 두께를 지정합니다. 기본값은 1입니다. 음수 값을 전달하면 내부를 채웁니다.
lineType: 선택적으로 선의 형태를 지정합니다. 기본값은 cv2.LINE_8입니다.
shift: 선택적으로 좌표값의 소수 부분을 비트 시프트할 양을 지정합니다.
"""
# cv2.waitKey(0)
# cv2.destroyAllWindows()
~~~

- 카메라 켈리브레이션 사진 촬영
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/541e57af-7bc3-40f3-9f81-7d343eb0d5f6" />
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/00c6ffa4-2e03-4ecb-bbfe-f8859954d251" />
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/1e643b11-43d2-4dfe-a1b6-a72a04d6fc14" />

~~~
import cv2 
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar
import webbrowser  # 웹사이트 열기 위한 모듈
import cv2
import datetime

cap = cv2.VideoCapture(0)
# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)
while True:

    for d in decoded:
        x, y, w, h = d.rect
        barcode_data = d.data.decode('utf-8')
        barcode_type = d.type
    # 카메라로부터 프레임을 읽음
    ret, frame = cap.read()
    if not ret:
        print("프레임 X")  # 프레임 읽기 실패 시 메시지 출력
        break

        text = '%s (%s)' % (barcode_data, barcode_type)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0 ), 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # 읽은 프레임을 화면에 표시
    cv2.imshow("Video", frame)

        # 'qt'가 들어 있으면 웹브라우저 열기 (한 번만)
        if 'http' in barcode_data.lower():  # 대소문자 구분 없이
            print(f"Opening URL: {barcode_data}")
            webbrowser.open(barcode_data)
            cap.release()
            cv2.destroyAllWindows()
            exit()
    # 키 입력을 기다림 (1ms 대기 후 다음 프레임으로 이동)
    key = cv2.waitKey(1) & 0xFF

    cv2.imshow('camera', img)
    # 'a' 키가 눌리면 현재 프레임을 저장
    if key == ord('a'):
        # 파일 이름을 현재 날짜 및 시간으로 설정
        filename = datetime.datetime.now().strftime("../img/capture_%Y%m%d_%H%M%S.png")
        # 프레임을 이미지 파일로 저장
        cv2.imwrite(filename, frame)
        print(f"{filename}")  # 저장된 파일 이름 출력

    key = cv2.waitKey(1)
    if key == ord('q'):
    # 'q' 키가 눌리면 루프를 종료
    elif key == ord('q'):
break

cap.release() 
cv2.destroyAllWindows()
# 자원 해제 (카메라 및 모든 OpenCV 창 닫기)
cap.release()
cv2.destroyAllWindows()
~~~

- 카메라 켈레브레이션 보정
~~~
import cv2
import datetime
import numpy as np
import os
import glob
import pickle

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0) 
def test_different_checkerboard_sizes(img_path):
    """다양한 체커보드 크기로 테스트해보는 함수"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 일반적인 체커보드 크기들
    checkerboard_sizes = [
        (7, 10), (10, 7),   # 원래 설정
        (6, 9), (9, 6),     # 8x10 체커보드
        (5, 8), (8, 5),     # 6x9 체커보드
        (4, 7), (7, 4),     # 5x8 체커보드
        (6, 8), (8, 6),     # 7x9 체커보드
        (5, 7), (7, 5),     # 6x8 체커보드
        (4, 6), (6, 4),     # 5x7 체커보드
        (3, 5), (5, 3),     # 4x6 체커보드
    ]
    
    print(f"\n=== {os.path.basename(img_path)} 체커보드 크기 테스트 ===")
    
    successful_sizes = []
    
    for size in checkerboard_sizes:
        ret, corners = cv2.findChessboardCorners(gray, size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print(f"✓ {size} 크기로 체커보드 검출 성공!")
            successful_sizes.append(size)
        else:
            print(f"✗ {size} 크기로 검출 실패")
    
    return successful_sizes

while True:
def analyze_image_quality(img_path):
    """이미지 품질 분석 함수"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {img_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\n=== {os.path.basename(img_path)} 이미지 분석 ===")
    print(f"이미지 크기: {img.shape[1]} x {img.shape[0]}")
    print(f"평균 밝기: {np.mean(gray):.1f}")
    print(f"밝기 표준편차: {np.std(gray):.1f}")
    
    # 대비 분석
    contrast = gray.max() - gray.min()
    print(f"대비: {contrast}")
    
    # 블러 정도 분석 (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"선명도 (높을수록 좋음): {laplacian_var:.1f}")
    
    if laplacian_var < 100:
        print("⚠️  이미지가 흐릿할 수 있습니다.")
    if contrast < 100:
        print("⚠️  이미지 대비가 낮습니다.")
    if np.mean(gray) < 50:
        print("⚠️  이미지가 너무 어둡습니다.")
    elif np.mean(gray) > 200:
        print("⚠️  이미지가 너무 밝습니다.")

    # 카메라로부터 프레임을 읽음
    ret, frame = cap.read()
    if not ret:
        print("프레임 X")  # 프레임 읽기 실패 시 메시지 출력
        break
def show_preprocessed_image(img_path, checkerboard_size=(7, 10)):
    """전처리된 이미지를 보여주는 함수"""
    img = cv2.imread(img_path)
    if img is None:
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 다양한 전처리 방법들
    # 1. 히스토그램 평활화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    
    # 2. 가우시안 블러
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. 이진화
    _, gray_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 적응적 이진화
    gray_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
    
    # 각각에 대해 체커보드 검출 시도
    methods = [
        ("Original", gray),
        ("CLAHE", gray_clahe),
        ("Gaussian Blur", gray_blur),
        ("Threshold", gray_thresh),
        ("Adaptive Threshold", gray_adaptive)
    ]
    
    print(f"\n=== {os.path.basename(img_path)} 전처리 방법별 테스트 ===")
    
    best_result = None
    best_method = None
    
    for method_name, processed_img in methods:
        ret, corners = cv2.findChessboardCorners(processed_img, checkerboard_size,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            print(f"✓ {method_name}: 체커보드 검출 성공!")
            if best_result is None:
                best_result = (processed_img, corners)
                best_method = method_name
        else:
            print(f"✗ {method_name}: 체커보드 검출 실패")
    
    # 결과 시각화
    if best_result is not None:
        processed_img, corners = best_result
        result_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(result_img, checkerboard_size, corners, True)
        
        # 이미지 크기 조정
        height, width = result_img.shape[:2]
        if height > 600 or width > 800:
            scale = min(600/height, 800/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            result_img = cv2.resize(result_img, (new_width, new_height))
        
        cv2.imshow(f'Best Result - {best_method}', result_img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    
    return best_result is not None

    # 읽은 프레임을 화면에 표시
    cv2.imshow("Video", frame)
def calibrate_camera_flexible():
    """유연한 체커보드 검출을 위한 캘리브레이션 함수"""
    
    # 다양한 이미지 형식과 경로 시도
    image_paths = [
        '../img/*.png', '../img/*.jpg', '../img/*.jpeg',
        './img/*.png', './img/*.jpg', './img/*.jpeg',
        'img/*.png', 'img/*.jpg', 'img/*.jpeg',
        '*.png', '*.jpg', '*.jpeg'
    ]
    
    images = []
    for path_pattern in image_paths:
        found_images = glob.glob(path_pattern)
        if found_images:
            images.extend(found_images)
    
    images = list(set(images))
    
    if not images:
        print("체커보드 이미지를 찾을 수 없습니다!")
        return None
    
    print(f"총 {len(images)}개의 이미지를 발견했습니다.")
    
    # 첫 번째 이미지로 체커보드 크기 자동 감지
    print("\n=== 체커보드 크기 자동 감지 ===")
    first_image = images[0]
    successful_sizes = test_different_checkerboard_sizes(first_image)
    
    if not successful_sizes:
        print("첫 번째 이미지에서 체커보드를 찾을 수 없습니다.")
        print("이미지 품질을 분석합니다...")
        analyze_image_quality(first_image)
        
        # 전처리 방법 테스트
        print("다양한 전처리 방법을 테스트합니다...")
        if show_preprocessed_image(first_image):
            print("전처리를 통해 검출이 가능할 수 있습니다.")
        
        return None
    
    # 가장 많이 검출된 크기 선택
    CHECKERBOARD = successful_sizes[0]
    print(f"선택된 체커보드 크기: {CHECKERBOARD}")
    
    # 캘리브레이션 진행
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    
    # 3D 점의 세계 좌표 정의
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    successful_detections = 0
    
    for i, fname in enumerate(images):
        print(f"처리 중: {os.path.basename(fname)} ({i+1}/{len(images)})")
        
        img = cv2.imread(fname)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 여러 전처리 방법 시도
        preprocessing_methods = [
            ("original", gray),
            ("clahe", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)),
            ("blur", cv2.GaussianBlur(gray, (3, 3), 0))
        ]
        
        corners_found = False
        for method_name, processed_gray in preprocessing_methods:
            ret, corners = cv2.findChessboardCorners(processed_gray, CHECKERBOARD,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_FAST_CHECK +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(processed_gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                successful_detections += 1
                corners_found = True
                
                print(f"  ✓ 체커보드 검출 성공 ({method_name})")
                
                # 결과 시각화
                img_corners = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
                height, width = img_corners.shape[:2]
                if height > 600 or width > 800:
                    scale = min(600/height, 800/width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img_corners = cv2.resize(img_corners, (new_width, new_height))
                
                cv2.imshow('Checkerboard Detection', img_corners)
                cv2.waitKey(300)
                break
        
        if not corners_found:
            print(f"  ✗ 체커보드 검출 실패")
    
    cv2.destroyAllWindows()
    
    print(f"\n총 {successful_detections}개 이미지에서 체커보드 검출 성공")
    
    if successful_detections < 3:
        print("캘리브레이션을 위해서는 최소 3개 이상의 성공적인 검출이 필요합니다.")
        return None
    
    # 카메라 캘리브레이션 수행
    print("카메라 캘리브레이션을 수행 중...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                      gray.shape[::-1], None, None)
    
    if ret:
        print("캘리브레이션 성공!")
        print("Camera matrix:")
        print(mtx)
        print("\nDistortion coefficients:")
        print(dist)
        
        calibration_data = {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'checkerboard_size': CHECKERBOARD
        }
        
        with open('camera_calibration.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print("캘리브레이션 데이터가 저장되었습니다.")
        return calibration_data
    else:
        print("캘리브레이션 실패!")
        return None

    # 키 입력을 기다림 (1ms 대기 후 다음 프레임으로 이동)
    key = cv2.waitKey(1) & 0xFF
def live_video_correction(calibration_data):
    """실시간 비디오 왜곡 보정"""
    if calibration_data is None:
        print("캘리브레이션 데이터가 없습니다.")
        return
    
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    print("실시간 왜곡 보정을 시작합니다. 'q'를 눌러 종료하세요.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        x, y, w_roi, h_roi = roi
        if all(v > 0 for v in [x, y, w_roi, h_roi]):
            dst = dst[y:y+h_roi, x:x+w_roi]
        
        try:
            original = cv2.resize(frame, (640, 480))
            corrected = cv2.resize(dst, (640, 480))
            combined = np.hstack((original, corrected))
            
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Corrected", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Camera Calibration Result', combined)
        except:
            cv2.imshow('Original', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # 'a' 키가 눌리면 현재 프레임을 저장
    if key == ord('a'):
        # 파일 이름을 현재 날짜 및 시간으로 설정
        filename = datetime.datetime.now().strftime("../img/capture_%Y%m%d_%H%M%S.png")
        # 프레임을 이미지 파일로 저장
        cv2.imwrite(filename, frame)
        print(f"{filename}")  # 저장된 파일 이름 출력

    # 'q' 키가 눌리면 루프를 종료
    elif key == ord('q'):
        break

# 자원 해제 (카메라 및 모든 OpenCV 창 닫기)
cap.release()
cv2.destroyAllWindows()
if __name__ == "__main__":
    print("=== 향상된 카메라 캘리브레이션 프로그램 ===")
    
    if os.path.exists('camera_calibration.pkl'):
        choice = input("기존 캘리브레이션 데이터를 사용하시겠습니까? (y/n): ")
        if choice.lower() == 'y':
            with open('camera_calibration.pkl', 'rb') as f:
                calibration_data = pickle.load(f)
        else:
            calibration_data = calibrate_camera_flexible()
    else:
        calibration_data = calibrate_camera_flexible()
    
    if calibration_data is not None:
        print("\n실시간 비디오 보정을 시작합니다...")
        live_video_correction(calibration_data)
    else:
        print("\n캘리브레이션에 실패했습니다.")
        print("다음 사항을 확인해보세요:")
        print("1. 체커보드가 명확하게 보이는 이미지인지 확인")
        print("2. 체커보드의 모든 코너가 이미지 안에 포함되어 있는지 확인")
        print("3. 이미지가 너무 흐리거나 어둡지 않은지 확인")
        print("4. 다양한 각도에서 촬영된 이미지들인지 확인")
~~~

- 켈레브레이션 ArucoMarker2
<img width="642" height="512" alt="아루코마커" src="https://github.com/user-attachments/assets/cdb425a2-b124-43c8-800e-a467edfdbc64" />

~~~
import cv2
import numpy as np
import os
import time
import pickle


def estimate_pose_single_marker(corners, marker_size, camera_matrix, dist_coeffs):
    """
    단일 마커의 포즈를 추정하는 함수 (OpenCV 4.7+ 호환)
    cv2.aruco.estimatePoseSingleMarkers의 대체 함수
    """
    # 마커의 3D 좌표 정의 (마커 중심을 원점으로)
    half_size = marker_size / 2
    object_points = np.array([
        [-half_size, half_size, 0],   
        [half_size, half_size, 0],    
        [half_size, -half_size, 0],   
        [-half_size, -half_size, 0]   
    ], dtype=np.float32)
    
    # 이미지 좌표 (2D)
    image_points = corners[0].astype(np.float32)
    
    # PnP 문제 해결
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    
    if success:
        return rvec, tvec
    else:
        return None, None


def live_aruco_detection(calibration_data):
    """
    실시간으로 비디오를 받아 ArUco 마커를 검출하고 3D 포즈를 추정하는 함수

    Args:
        calibration_data: 카메라 캘리브레이션 데이터를 포함한 딕셔너리
            - camera_matrix: 카메라 내부 파라미터 행렬
            - dist_coeffs: 왜곡 계수
    """
    # 캘리브레이션 데이터 추출
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']

    # ArUco 검출기 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # 마커 크기 설정 (미터 단위)
    marker_size = 0.05  # 예: 5cm = 0.05m

    # 카메라 설정
    cap = cv2.VideoCapture(0)

    # 카메라 초기화 대기
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 이미지 왜곡 보정
        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # 마커 검출
        corners, ids, rejected = detector.detectMarkers(frame_undistorted)

        # 마커가 검출되면 표시 및 포즈 추정
        if ids is not None:
            # 검출된 마커 표시
            cv2.aruco.drawDetectedMarkers(frame_undistorted, corners, ids)

            # 각 마커에 대해 처리
            for i in range(len(ids)):
                # 포즈 추정 (새로운 방법으로 대체)
                rvec, tvec = estimate_pose_single_marker(
                    [corners[i]], marker_size, camera_matrix, dist_coeffs
                )
                
                if rvec is not None and tvec is not None:
                    # 좌표축 표시
                    cv2.drawFrameAxes(frame_undistorted, camera_matrix, dist_coeffs,
                                      rvec, tvec, marker_size/2)

                    # 마커의 3D 위치 표시
                    pos_x = tvec[0][0]
                    pos_y = tvec[1][0]
                    pos_z = tvec[2][0]

                    # 회전 벡터를 오일러 각도로 변환
                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    euler_angles = cv2.RQDecomp3x3(rot_matrix)[0]

                    # 마커 정보 표시
                    corner = corners[i][0]
                    center_x = int(np.mean(corner[:, 0]))
                    center_y = int(np.mean(corner[:, 1]))

                    cv2.putText(frame_undistorted,
                                f"ID: {ids[i][0]}",
                                (center_x, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 2)

                    cv2.putText(frame_undistorted,
                                f"Pos: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})m",
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 2)

                    cv2.putText(frame_undistorted,
                                f"Rot: ({euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f})deg",
                                (center_x, center_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 2)
                    
                    # 코너 포인트 표시
                    for point in corner:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(frame_undistorted, (x, y), 4, (0, 0, 255), -1)

        # 프레임 표시
        cv2.imshow('ArUco Marker Detection', frame_undistorted)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()


def main():
    # 캘리브레이션 데이터 로드
    try:
        with open('camera_calibration.pkl', 'rb') as f:
            calibration_data = pickle.load(f)
        print("Calibration data loaded successfully")
    except FileNotFoundError:
        print("Error: Camera calibration file not found")
        return
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return

    print("Starting ArUco marker detection...")
    live_aruco_detection(calibration_data)


if __name__ == "__main__":
    main()
~~~

- 거리에 따른 경고 메세지 출력
<img width="642" height="512" alt="캡처" src="https://github.com/user-attachments/assets/79737ffe-be73-44f7-8bb0-ad48a2a09f0c" />

~~~
(center_x, center_y + 20),
cv2.FONT_HERSHEY_SIMPLEX,
0.5, (255, 0, 255), 2)

                    #----------------------------------------------------------- 조건문
                    # 마커와 카메라 간의 거리가 30cm 이하인 경우 "STOP!" 메시지 표시
                    if pos_z < 0.30:  # 30cm 이하일 때
                        # 텍스트 배경 그리기 (배경 색상은 반투명 빨간색)
                        text = "STOP!"
                        text_size, _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_width, text_height = text_size

                        # 배경 사각형 그리기
                        background_x1 = center_x - text_width // 2 - 10
                        background_y1 = center_y + 40 - text_height // 2 - 10
                        background_x2 = center_x + text_width // 2 + 10
                        background_y2 = center_y + 40 + text_height // 2 + 10
                        cv2.rectangle(frame_undistorted, (background_x1, background_y1),
                                      (background_x2, background_y2), (0, 0, 255), -1)  # 빨간색 배경

                        # 텍스트 그리기
                        cv2.putText(frame_undistorted,
                                    text,
                                    (center_x - text_width // 2,
                                     center_y + 40 + text_height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 255, 255), 2, cv2.LINE_AA)  # 흰색 텍스트
                    
                    else:
                        text = "GO!"
                        text_size, _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_width, text_height = text_size

                        # 배경 사각형 그리기
                        background_x1 = center_x - text_width // 2 - 10
                        background_y1 = center_y + 40 - text_height // 2 - 10
                        background_x2 = center_x + text_width // 2 + 10
                        background_y2 = center_y + 40 + text_height // 2 + 10
                        cv2.rectangle(frame_undistorted, (background_x1, background_y1),
                                      (background_x2, background_y2), (0, 255, 0), -1)  # 초록색 배경

                        # 텍스트 그리기
                        cv2.putText(frame_undistorted,
                                    text,
                                    (center_x - text_width // 2,
                                     center_y + 40 + text_height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 255, 255), 2, cv2.LINE_AA)  # 흰색 텍스트

#코너 포인트 표시
for point in corner:

~~~
  
