#!/usr/bin/env python
# -*- coding: utf-8 -*-
from this import s
from tkinter import Scale
from sklearn.ensemble import RandomForestClassifier
import collections
from collections import deque
import mediapipe as mp
import numpy as np
import copy
import argparse
import random
from playsound import playsound
import pyautogui as pgui
import os
import time
import pyperclip
import keyboard

import cv2 as cv
print('cv2', cv.__version__)


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded


def get_args():
    parser = argparse.ArgumentParser()

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


model = None
model_count = 0
X = []
Y = []


def update_model(X, y):
    global model
    print('Updating model datapoints = ', len(y))
    model = RandomForestClassifier()
    model.fit(X, y)


window_size = 3
buffer = collections.deque(maxlen=window_size)


def init_buffer():
    buffer.clear()


def transform(hand_landmarks, mode):
    global model, model_count
    hand_landmarks = hand_landmarks[0]  # これは何？
    datapoint = []
    for landmark in hand_landmarks.landmark:
        datapoint.append(landmark.x)
        datapoint.append(landmark.y)
        datapoint.append(landmark.z)
    # データポイントをnumpyに変換
    datapoint = np.array(datapoint)
    buffer.append(datapoint)
    if len(buffer) != window_size:
        return 0
    window = np.concatenate(list(buffer), 0)
    if mode != 0:
        X.append(window)
        Y.append(mode)
        print(mode, window)
    else:
        if len(Y) != model_count:
            update_model(X, Y)
            model_count = len(Y)
        if model is not None:
            y_pred = model.predict([window])
            print(y_pred)
            return y_pred[0]
    return 0


StartKey = ord('s')
PositiveKey = ord('p')
NegativeKey = ord('n')
PrettyTricKey = ord('w')
Trick= ord('t')
Sound = ord('y')
'''
T_=['t','r','i','c','k']
for i in T_:
    Trick.append(ord(i))
    '''
#Trick = ['b','o','o']


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    mode = 0
    mode_s = ''

    #気分の値feelとする
    feel=0
    #画像読み込み
    #b = random.randrange(10)
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
    #img6 = cv.imread('./img/hart1_70px.png')
    #img6 = cv.resize(img6 , (iwidth, iheight))
    #img7 = cv.imread('./img/hart2_70px.png')
    #img7 = cv.resize(img7 , (iwidth, iheight))
    
    #エンディング画像
    gwidth=700
    gheight=500
    end_p1 = cv.imread('./img/end_p1.JPG')
    end_p1 = cv.resize(end_p1 , (gwidth, gheight))
    end_p2 = cv.imread('./img/end_p2.JPG')
    end_p2 = cv.resize(end_p2 , (gwidth, gheight))
    end_p3 = cv.imread('./img/end_p3.JPG')
    end_p3 = cv.resize(end_p3 , (gwidth, gheight))
    end_p4 = cv.imread('./img/end_p4.JPG')
    end_p4 = cv.resize(end_p4 , (gwidth, gheight))
    end_p5 = cv.imread('./img/end_p5.JPG')
    end_p5 = cv.resize(end_p5 , (gwidth, gheight))
    end_n1 = cv.imread('./img/end_n1.JPG')
    end_n1 = cv.resize(end_n1 , (gwidth, gheight))
    end_n2 = cv.imread('./img/end_n2.JPG')
    end_n2 = cv.resize(end_n2 , (gwidth, gheight))
    end_n3 = cv.imread('./img/end_n3.JPG')
    end_n3 = cv.resize(end_n3 , (gwidth, gheight))
    end_n4 = cv.imread('./img/end_n4.JPG')
    end_n4 = cv.resize(end_n4 , (gwidth, gheight))
    
    end_p=[end_p1,end_p2,end_p3,end_p4,end_p5]
    end_n=[end_n1,end_n2,end_n3,end_n4]

    #やばいゾンビちゃん画像(この画像が出てくるKeyの名前をPrettyTricKeyにしてるのでPtという名前)
    jwidth=350
    jheight=350
    Pt_1 = cv.imread('./img/zomG_a2.png')
    Pt_1 = cv.resize(Pt_1 , (jwidth, jheight))
    Pt_2 = cv.imread('./img/zomG_b1.png')
    Pt_2 = cv.resize(Pt_2 , (jwidth, jheight))
    Pt_3 = cv.imread('./img/zomG_b2.png')
    Pt_3 = cv.resize(Pt_3 , (jwidth, jheight))
    Pt_4 = cv.imread('./img/zomG_c12.png')
    Pt_4 = cv.resize(Pt_4 , (jwidth, jheight))
    Pt_5 = cv.imread('./img/zomG_d0.png')
    Pt_5 = cv.resize(Pt_5 , (jwidth, jheight))
    Pt_6 = cv.imread('./img/zomG_d2.png')
    Pt_6 = cv.resize(Pt_6 , (jwidth, jheight))
    Pt_7 = cv.imread('./img/zomG_d3.png')
    Pt_7 = cv.resize(Pt_7 , (jwidth, jheight))
    Pt_8 = cv.imread('./img/zomG_d6.png')
    Pt_8 = cv.resize(Pt_8 , (jwidth, jheight))
    Pt_9 = cv.imread('./img/zomG_d10.png')
    Pt_9 = cv.resize(Pt_9 , (jwidth, jheight))
    Pt_10 = cv.imread('./img/zomG_d12.png')
    Pt_10 = cv.resize(Pt_10 , (jwidth, jheight))

    Pt_list = [Pt_1, Pt_2, Pt_3, Pt_4, Pt_5, Pt_6, Pt_7, Pt_8, Pt_9, Pt_10]

    #お菓子画像
    kwidth=50
    kheight=50
    Sw_1 = cv.imread('./img/sweets/beans1.png')
    Sw_1 = cv.resize(Sw_1 , (kwidth, kheight))
    Sw_2 = cv.imread('./img/sweets/cake1.png')
    Sw_2 = cv.resize(Sw_2 , (kwidth, kheight))
    Sw_3 = cv.imread('./img/sweets/candies1.png')
    Sw_3 = cv.resize(Sw_3 , (kwidth, kheight))
    Sw_4 = cv.imread('./img/sweets/cho1.png')
    Sw_4 = cv.resize(Sw_4 , (kwidth, kheight))
    Sw_5 = cv.imread('./img/sweets/cookie1.png')
    Sw_5 = cv.resize(Sw_5 , (kwidth, kheight))
    Sw_6 = cv.imread('./img/sweets/cookie2.png')
    Sw_6 = cv.resize(Sw_6 , (kwidth, kheight))
    Sw_7 = cv.imread('./img/sweets/cookie3.png')
    Sw_7 = cv.resize(Sw_7 , (kwidth, kheight))
    Sw_8 = cv.imread('./img/sweets/cookie4.png')
    Sw_8 = cv.resize(Sw_8 , (kwidth, kheight))
    Sw_9 = cv.imread('./img/sweets/cookie5.png')
    Sw_9 = cv.resize(Sw_9 , (kwidth, kheight))
    Sw_10 = cv.imread('./img/sweets/cookie6.png')
    Sw_10 = cv.resize(Sw_10 , (kwidth, kheight))
    Sw_11 = cv.imread('./img/sweets/cookie7.png')
    Sw_11 = cv.resize(Sw_11 , (kwidth, kheight))
    Sw_12 = cv.imread('./img/sweets/cookie7.png')
    Sw_12 = cv.resize(Sw_12 , (kwidth, kheight))
    Sw_13 = cv.imread('./img/sweets/donut1.png')
    Sw_13 = cv.resize(Sw_13 , (kwidth, kheight))
    Sw_14 = cv.imread('./img/sweets/donut2.png')
    Sw_14 = cv.resize(Sw_14 , (kwidth, kheight))
    Sw_15 = cv.imread('./img/sweets/donut3.png')
    Sw_15 = cv.resize(Sw_15 , (kwidth, kheight))
    Sw_16 = cv.imread('./img/sweets/maro1.png')
    Sw_16 = cv.resize(Sw_16 , (kwidth, kheight))
    Sw_17 = cv.imread('./img/sweets/maro2.png')
    Sw_17 = cv.resize(Sw_17 , (kwidth, kheight))
    Sw_18 = cv.imread('./img/sweets/pero1.png')
    Sw_18 = cv.resize(Sw_18 , (kwidth, kheight))
    Sw_19 = cv.imread('./img/sweets/pero2.png')
    Sw_19 = cv.resize(Sw_19 , (kwidth, kheight))
    Sw_20 = cv.imread('./img/sweets/ron1.png')
    Sw_20 = cv.resize(Sw_20 , (kwidth, kheight))
    Sw_21 = cv.imread('./img/sweets/ron2.png')
    Sw_21 = cv.resize(Sw_21 , (kwidth, kheight))
    Sw_22 = cv.imread('./img/sweets/spider1.png')
    Sw_22 = cv.resize(Sw_22 , (kwidth, kheight))
    Sw_23 = cv.imread('./img/sweets/tou1.png')
    Sw_23 = cv.resize(Sw_23 , (kwidth, kheight))

    Sw_list = [Sw_1, Sw_2, Sw_3, Sw_4, Sw_5, Sw_6, Sw_7, Sw_8, Sw_9, Sw_10, Sw_11, Sw_12, Sw_13, Sw_14, Sw_15, Sw_16, Sw_17, Sw_18, Sw_19, Sw_20, Sw_21, Sw_22, Sw_23]

    #お化けっぽいやつら
    Gh_1 = cv.imread('./img/ghost/bat1.png')
    Gh_1 = cv.resize(Gh_1 , (kwidth, kheight))
    Gh_2 = cv.imread('./img/ghost/bat2.png')
    Gh_2 = cv.resize(Gh_2 , (kwidth, kheight))
    Gh_3 = cv.imread('./img/ghost/bat3.png')
    Gh_3 = cv.resize(Gh_3 , (kwidth, kheight))
    Gh_4 = cv.imread('./img/ghost/bat4.png')
    Gh_4 = cv.resize(Gh_4 , (kwidth, kheight))
    Gh_5 = cv.imread('./img/ghost/bat5.png')
    Gh_5 = cv.resize(Gh_5 , (kwidth, kheight))
    Gh_6 = cv.imread('./img/ghost/bat6.png')
    Gh_6 = cv.resize(Gh_6 , (kwidth, kheight))
    Gh_7 = cv.imread('./img/ghost/born1.png')
    Gh_7 = cv.resize(Gh_7 , (kwidth, kheight))
    Gh_8 = cv.imread('./img/ghost/born2.png')
    Gh_8 = cv.resize(Gh_8 , (kwidth, kheight))
    Gh_9 = cv.imread('./img/ghost/born3.png')
    Gh_9 = cv.resize(Gh_9 , (kwidth, kheight))
    Gh_10 = cv.imread('./img/ghost/cat1.png')
    Gh_10 = cv.resize(Gh_10 , (kwidth, kheight))
    Gh_11 = cv.imread('./img/ghost/cat2.png')
    Gh_11 = cv.resize(Gh_11 , (kwidth, kheight))
    Gh_12 = cv.imread('./img/ghost/cat3.png')
    Gh_12 = cv.resize(Gh_12 , (kwidth, kheight))
    Gh_13 = cv.imread('./img/ghost/face1.png')
    Gh_13 = cv.resize(Gh_13 , (kwidth, kheight))
    Gh_14 = cv.imread('./img/ghost/face2.png')
    Gh_14 = cv.resize(Gh_14 , (kwidth, kheight))
    Gh_15 = cv.imread('./img/ghost/face4.png')
    Gh_15 = cv.resize(Gh_15 , (kwidth, kheight))
    Gh_16 = cv.imread('./img/ghost/face5.png')
    Gh_16 = cv.resize(Gh_16 , (kwidth, kheight))
    Gh_17 = cv.imread('./img/ghost/ghost1.png')
    Gh_17 = cv.resize(Gh_17 , (kwidth, kheight))
    Gh_18 = cv.imread('./img/ghost/ghost2.png')
    Gh_18 = cv.resize(Gh_18 , (kwidth, kheight))
    Gh_19 = cv.imread('./img/ghost/ghost3.png')
    Gh_19 = cv.resize(Gh_19 , (kwidth, kheight))
    Gh_20 = cv.imread('./img/ghost/ghost4.png')
    Gh_20 = cv.resize(Gh_20 , (kwidth, kheight))
    Gh_21 = cv.imread('./img/ghost/ghost5.png')
    Gh_21 = cv.resize(Gh_21 , (kwidth, kheight))
    Gh_22 = cv.imread('./img/ghost/grave1.png')
    Gh_22 = cv.resize(Gh_22 , (kwidth, kheight))
    Gh_23 = cv.imread('./img/ghost/grave2.png')
    Gh_23 = cv.resize(Gh_23 , (kwidth, kheight))
    Gh_24 = cv.imread('./img/ghost/pum1.png')
    Gh_24 = cv.resize(Gh_24 , (kwidth, kheight))
    Gh_25 = cv.imread('./img/ghost/pum2.png')
    Gh_25 = cv.resize(Gh_25 , (kwidth, kheight))

    Gh_list=[Gh_1, Gh_2, Gh_3, Gh_4, Gh_5, Gh_6, Gh_7, Gh_8, Gh_9, Gh_10, Gh_11, Gh_13, Gh_14, Gh_15, Gh_16, Gh_17, Gh_18, Gh_19, Gh_20, Gh_21, Gh_22, Gh_23, Gh_24, Gh_25]

    #コマンドt
    t_1 = cv.imread('./img/command/hands1.png')
    t_1 = cv.resize(t_1 , (600, 400))
    t_2 = cv.imread('./img/command/hand2.png')
    t_2 = cv.resize(t_2 , (300, 300))
    t_3 = cv.imread('./img/command/hand3.png')
    t_3 = cv.resize(t_3 , (300, 300))
    t_4 = cv.imread('./img/command/mask1.png')
    t_4 = cv.resize(t_4 , (300, 300))
    t_5 = cv.imread('./img/command/mask2.png')
    t_5 = cv.resize(t_5 , (300, 300))

    t_list=[t_1,t_3,t_5]

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ  #####################################################
        ret, image = cap.read() #imageは写ってる人の画面
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        # 描画 ################################################################
        #res = transform(hand_landmarks, mode)
        #res = transform(hand_landmarks, mode)
        #feel+=res

        if results.multi_hand_landmarks is None:
            init_buffer()
        else:
            res = transform(results.multi_hand_landmarks, mode)
            feel+=res*10
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):              
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # 描画
                debug_image = draw_landmarks(debug_image, cx, cy,
                                             hand_landmarks, handedness)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                b = random.randrange(30)
                if feel>=100 and feel<300:
                    debug_image[0:iheight, 0:iwidth] = img2
                    s=5
                    debug_image = Sweets(debug_image, s)
                elif feel>=300 and feel<=500:
                    debug_image[0:iheight, 0:iwidth] = img2
                    s=10
                    debug_image = Sweets(debug_image, s)
                    if b==21:
                        playsound("./se/twinkle1.mp3")
                elif feel>=500 and feel<1200:
                    debug_image[0:iheight, 0:iwidth] = img3
                    s=18
                    debug_image = Sweets(debug_image, s)
                    if b==1:
                        playsound("./se/twinkle1.mp3")
                #if feel>=600 and feel<800:
                    #debug_image[20:iheight, 20:iwidth] = img6
                #elif feel>=800:
                    #debug_image[20:iheight, 20:iwidth] = img6
                    #debug_image[50:iheight, 50:iwidth] = img7
                elif feel>=1300 and feel<1350:
                    debug_image[0:iheight, 0:iwidth] = img3
                    s=17
                    debug_image = Sweets(debug_image, s)
                    e_p = random.randrange(5)
                    cv.imshow('Happy ending!!', end_p[e_p])
                    #debug_image[0:iheight, 0:iwidth] = end_p[e_p]
                    #feel=0j
                elif feel >=1350:
                    feel=0
                elif feel<=-100 and feel>-300:
                    debug_image[0:iheight, 0:iwidth] = img4
                    g=5
                    debug_image = Ghost(debug_image, g)
                    if b==2:
                        playsound("./se/failed1.mp3")
                elif feel<=-300 and feel>-500:
                    debug_image[0:iheight, 0:iwidth] = img4
                    g=10
                    debug_image = Ghost(debug_image, g)
                elif feel<=-500 and feel>-1200:
                    debug_image[0:iheight, 0:iwidth] = img5
                    g=18
                    debug_image = Ghost(debug_image, g)
                    if b==15:
                        playsound("./se/failed2.mp3")
                    '''
                    #やばいゾンビちゃん中央表示
                    p = random.randrange(10)
                    Pt = Pt_list[p]
                    debug_image = Paste(debug_image,Pa)
                    '''
                   # if b%2==0:
                     #   playsound("./se/failed2.mp3")
                elif feel<=-1300 and feel>-1350:
                    debug_image[0:iheight, 0:iwidth] = img5
                    g=18
                    debug_image = Ghost(debug_image, g)
                    e_n = random.randrange(4)
                    cv.imshow('Bad ending...', end_n[e_n])
                    #debug_image[0:iheight, 0:iwidth] = end_n[e_n]
                    #feel=0
                elif feel<=-1350:
                    feel=0
                else:
                    debug_image[0:iheight,0:iwidth] = img1
                cv.putText(debug_image, "Feel:" + str(feel), (10, 260),
                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                

        cv.putText(debug_image, "FPS:" + str(display_fps)+mode_s, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        #超便利: リアルタイム画像(debug_image)に画像貼り付け関数
        def cvpaste(img, imgback, x, y, angle, scale):  
            # x and y are the distance from the center of the background image 

            r = img.shape[0]
            c = img.shape[1]
            rb = imgback.shape[0]
            cb = imgback.shape[1]
            hrb=round(rb/2)
            hcb=round(cb/2)
            hr=round(r/2)
            hc=round(c/2)

            # Copy the forward image and move to the center of the background image
            imgrot = np.zeros((rb,cb,3),np.uint8)
            imgrot[hrb-hr:hrb+hr,hcb-hc:hcb+hc,:] = img[:hr*2,:hc*2,:]

            # Rotation and scaling
            M = cv.getRotationMatrix2D((hcb,hrb),angle,scale)
            imgrot = cv.warpAffine(imgrot,M,(cb,rb))
            # Translation
            M = np.float32([[1,0,x],[0,1,y]])
            imgrot = cv.warpAffine(imgrot,M,(cb,rb))

            # Makeing mask
            imggray = cv.cvtColor(imgrot,cv.COLOR_BGR2GRAY)
            ret, mask = cv.threshold(imggray, 10, 255, cv.THRESH_BINARY)
            mask_inv = cv.bitwise_not(mask)

            # Now black-out the area of the forward image in the background image
            img1_bg = cv.bitwise_and(imgback,imgback,mask = mask_inv)

            # Take only region of the forward image.
            img2_fg = cv.bitwise_and(imgrot,imgrot,mask = mask)

            # Paste the forward image on the background image
            imgpaste = cv.add(img1_bg,img2_fg)
            return imgpaste

        #ランダムな画像を貼り付ける関数
        def Paste(debug_image,Pa):
            x=random.randint(-300,300)
            y=random.randint(-300,300)
            angle = random.randint(0,360)
            scale = random.uniform(0.5,1.5)
            return cvpaste(Pa, debug_image, x, y, angle, scale)
        
        #定位置
        def Paste2(debug_image,Pt):
            x=0
            y=0
            angle = 0
            scale = 1.5
            return cvpaste(Pt, debug_image, x, y, angle, scale)

        #Sweetsランダム貼り付け
        def Sweets(debug_image, s):
            img_d = debug_image
            for i in range(s):
                a = random.randrange(22)
                Sw = Sw_list[a]
                Sw_pa = Paste(img_d, Sw)
                img_d = Sw_pa
            return img_d

        #Ghostランダム貼り付け
        def Ghost(debug_image, g):
            img_d = debug_image
            for i in range(g):
                a = random.randrange(24)
                Gh = Gh_list[a]
                Gh_pa = Paste(img_d, Gh)
                img_d = Gh_pa
            return img_d

        #ランダムワード抽出
        def words():
            word = ['Happy Halloween!!','Boo!!','\'I\'m so cute♥','You\'re so cute♥','I love you!!','All is well!','You can do that!']
            w = random.choice(word)
            emoji = ['😍', '🥰', '😋','👻','💋','💖','🎃','🧟']
            e = random.choice(emoji)
            return w,e

        #メモ帳開いて入力
        def kawaii():
            os.startfile(r'C:\Users\misai\mysite\mejiro_fes\Happy_Halloween.txt')
            time.sleep(1)
            w = words()
            pyperclip.copy(w[1])
            pgui.write(w[0])
            pgui.hotkey("ctrl","v")
            '''
            pgui.click(50,50)
            pgui.typewrite('Hello, pyautogui key!')
            '''

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == PositiveKey:
            mode_s = 'Positive'
            init_buffer()
            mode = 1
        elif key == NegativeKey:
            mode_s = 'Negative'
            init_buffer()
            mode = -1
        elif key == StartKey:
            mode_s = ''
            init_buffer()
            mode = 0
        elif key == 13: # Enter
            p = random.randrange(10)
            Pt = Pt_list[p]
            cv.imshow('PrettyTrick♥',Pt)
        elif key == PrettyTricKey:
            kawaii()
        elif key == Trick:
            p = random.randrange(3)
            Pt = t_list[p]
            debug_image = Paste2(debug_image,Pt)
        elif key == Sound:
            se = ['./se/man1.mp3','./se/man2.mp3','./se/door1.mp3']
            s = random.choice(se)
            playsound(s)

            '''
            a=1
            for i in range(1,5):
                if key == Trick[i]:
                    a+=1
                elif a==5:
                    kawaii()
                else:
                    a=1
                    break
                '''
            '''
            i=1
            T = []
            T.append('t')
            while True:
                if keyboard.read_key() == 'r':
                    T.append(T_[i])
                    i+=1
                elif T==T_:
                    p = random.randrange(10)
                    Pt = Pt_list[p]
                    cv.imshow('PrettyTrick♥',Pt)
                else:
                    T=[]
                    break
                '''

            #imgpaste = PtPaste(image,Pt)
            #cv.imshow('imgpaste',imgpaste)
#PrettyTricKey

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
