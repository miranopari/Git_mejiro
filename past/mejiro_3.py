#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
from hashlib import shake_128
from this import d
#from lib2to3.pytree import _Results
from unittest import result

import cv2 as cv #openCV
import numpy as np
import mediapipe as mp

import numpy as np
import pandas as pd

import pickle

import random

from os import path
from playsound import playsound

from utils import CvFpsCalc



def get_args():
    parser = argparse.ArgumentParser()
    #画面の大きさ設定
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args

#スタブ：関数座標渡して値戻す
def json_transform(hand_landmarks):
  hand_landmarks = hand_landmarks[0]
  record = {}
  for i, landmark in enumerate(hand_landmarks.landmark):
        record[str(i) + '_x'] = landmark.x
        record[str(i) + '_y'] = landmark.y
        record[str(i) + '_z'] = landmark.z

  df = pd.json_normalize(record)
  return df
#df = json_transform(hand_landmarks)

def panML(df):    
    X_test=df[df.columns].values
    print(f'X_test={X_test}')
    # 保存したモデルをロードする
    loaded_model = pickle.load(open("./finalized_model.sav", 'rb'))
    # loaded_model = pickle.load(open("<ここにモデルのパスを入れる>", 'rb'))
    y_pred = loaded_model.predict(X_test)
    a = random.randrange(10)
    print(f'y_pred[0]={y_pred[0]},{y_pred[0]+a}')
    if y_pred[0]==1:
        return -1
    else:
        return 1
    
'''
    a = random.randrange(10)
    if a%2==0:
        return 50
    else:
        return -50
        '''
        
    
    
        
        
        
"""
    #ファイルの名前をtrain.csvにするのを忘れずに！！！！！！！
    a = path.join(path.dirname(__file__), 'train_old.csv')
    #with open(a, 'r', encoding="utf-8") as f:
    s = f.read() #.splitlines()
    df = pd.read_csv(s)
    df.drop('movie_name',axis=1,inplace=True)
    df.drop('file_name',axis=1,inplace=True)
    # ランダムフォレスト
    from sklearn.model_selection import train_test_split
    X= df[df.columns[:-1]].values
    y = df["p/n"].values
    #訓練データとテストデータに分ける
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    from sklearn.ensemble import RandomForestClassifier
    #ランダムフォレストの作成
    d_tree = RandomForestClassifier()
    #fit()で学習させる。第一引数に説明変数、第二引数に目的変数
    d_tree = d_tree.fit(X_train, y_train)
    # ここに取ってきたデータいれる
    X_test = results
    #モデルから予測されたデータ列を代入
    y_pred = d_tree.predict(X_test)

    if y_pred[0]==0:
        return 1
    else:
        return -1
         """
    
    
    #print(y_pred[0])

# # 正解率を出す場合
# result = loaded_model.score(X_test, y_test)
# print(result)
    #if label==0:
        
    #else:
     #   return -1
    #results.multi_hand_landmarksを学習する
    # print(results.multi_hand_landmarks)
    #return 1 #1か-1返す

def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    #気分の値feelとする
    feel=0
    #画像読み込み
    iwidth=200
    iheight=200
    img1 = cv.imread('./img/zomG1_base.png')
    img1 = cv.resize(img1 , (iwidth, iheight))
    img2 = cv.imread('./img/zomG2_smile1.png')
    img2 = cv.resize(img2 , (iwidth, iheight))
    img3 = cv.imread('./img/zomG3_smile2.png')
    img3 = cv.resize(img3 , (iwidth, iheight))
    img4 = cv.imread('./img/zomG4_sad1.png')
    img4 = cv.resize(img4 , (iwidth, iheight))
    img5 = cv.imread('./img/zomG5_sad2.png')
    img5 = cv.resize(img5 , (iwidth, iheight))
    img6 = cv.imread('./img/hart1_70px.png')
    img6 = cv.resize(img6 , (iwidth, iheight))
    img7 = cv.imread('./img/hart2_70px.png')
    img7 = cv.resize(img7 , (iwidth, iheight))
    #img6 = cv.imread('./img/closing_eyes2.png')
    #img6 = cv.resize(img6 , (iwidth, iheight))
    #vid1 = cv.VideoCapture('closing_eyesG3.gif')
    '''
    gif1 = cv.VideoCapture('./gif/smilingG3.gif')
    fps1 = gif1.get(cv.CAP_PROP_FPS)  # fpsは１秒あたりのコマ数
    images1 = []
    i = 0
    while True:
        is_success, img_a = gif1.read()
        if not is_success:
            break
        images1.append(img_a)
        i += 1
    '''

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)
        
        
        
        

        # 描画 (残す)##############################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                df=json_transform(results.multi_hand_landmarks)
                res=panML(df)
                feel+=res
        #犬表示
                b = random.randrange(10)

                if feel>=100 and feel<500:
                    debug_image[0:iheight, 0:iwidth] = img2
            
                if b%2==0:
                    playsound("./se/twinkle1.mp3")
                elif feel>=500 and feel<1200:
                    debug_image[0:iheight, 0:iwidth] = img3
                    if b%2==0:
                        playsound("./se/twinkle2.mp3")
                if feel>=600 and feel<800:
                    debug_image[20:iheight, 20:iwidth] = img6
                elif feel>=800:
                    debug_image[20:iheight, 20:iwidth] = img6
                    debug_image[50:iheight, 50:iwidth] = img7
                elif feel>=1300:
                    feel=0
                elif feel<=-100 and feel>-500:
                    debug_image[0:iheight, 0:iwidth] = img4
                    if b%2==0:
                        playsound("./se/failed1.mp3")
                elif feel<=-500 and feel>-1200:
                    debug_image[0:iheight, 0:iwidth] = img5
                    if b%2==0:
                        playsound("./se/failed2.mp3")
                elif feel<=-1300:
                    feel=0
                else:
                    debug_image[0:iheight, 0:iwidth] = img1
        
        #feel表示
                cv.putText(debug_image, "Feel:" + str(feel), (10, 260),
                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                #print(results.multi_hand_landmarks)
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # 描画
                debug_image = draw_landmarks(debug_image, cx, cy,
                                             hand_landmarks, handedness)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 230),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Hand Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        #print(landmark_point)
        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, cx, cy, landmarks, handedness):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # 人差指
        cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # 中指
        cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # 薬指
        cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # 小指
        cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # 手の平
        cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        # handedness.classification[0].index
        # handedness.classification[0].score

        cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv.putText(image, handedness.classification[0].label[0],
                   (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                   2, cv.LINE_AA)  # label[0]:一文字目だけ

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    main()
