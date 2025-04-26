# color_vector/estimator.py

import numpy as np
import cv2
from scipy.optimize import fmin
from .independence import f_burel
from .utils import select_file
import os

def estimate_color_vectors_from_files():
    """
    ユーザーが画像ファイルを手動で2枚選択して、推定を行う。
    """
    NShadow_path = select_file("影のない画像（noshadow）を選択してください")
    SkinImage_path = select_file("色素成分の多い画像（regularSkin）を選択してください")

    if not NShadow_path or not SkinImage_path:
        raise ValueError("両方の画像ファイルを選択してください。")

    return color_vector_estimation(NShadow_path, SkinImage_path)

def color_vector_estimation(Nshadow_path,SkinImage_path):
    """
    入力した画像を読み込み、メラニン・ヘモグロビンのベクトルを求める
    入力　Nshadow_path：影のない小領域画像へのパス
    　　　SkinImage_path：色素成分が豊富にある画像へのパス
    """
    # # # ===============================================================================
    # # # 影のない小領域画像における肌平面を求める
    # # # ===============================================================================
    NShadow_rgb = cv2.imread(Nshadow_path)
    Nheight , Nwidth,Nchannel = NShadow_rgb.shape
    Nshadow_r = NShadow_rgb[:,:,2]
    Nshadow_g = NShadow_rgb[:,:,1]
    Nshadow_b = NShadow_rgb[:,:,0]
    # # # 画像空間から濃度空間へ変換
    NShadow_logr = -np.log(Nshadow_r/255)
    NShadow_logg = -np.log(Nshadow_g/255)
    NShadow_logb = -np.log(Nshadow_b/255)
    Nskin = np.array([NShadow_logr.flatten(),NShadow_logg.flatten(),NShadow_logb.flatten()])
    
    # # 濃度ベクトルSの固有値と固有ベクトルを計算    
    N_covariance = np.cov(Nskin)
    N_eig, N_eigvec = np.linalg.eig(N_covariance)
    Nidx = np.argsort(N_eig)[::-1]
    N_eigvec_T = N_eigvec.T 
    
    N_eigvec_sorted =N_eigvec_T[Nidx]
    N_pca1 = N_eigvec_sorted[0,:] 
    N_pca2 = N_eigvec_sorted[1,:] 

    # # ===============================================================================
    # # 色素成分が沢山ある画像の情報を影のない画像平面に射影する
    # # ===============================================================================
    rgb = cv2.imread(SkinImage_path)
    height ,width,channel = rgb.shape
    r = rgb[:,:,2]
    g = rgb[:,:,1]
    b = rgb[:,:,0]
    # # 画像空間から濃度空間へ変換
    # # 肌色ベクトル
    logr = -np.log(r/255)
    logg = -np.log(g/255)
    logb = -np.log(b/255)
    Skin = np.array([logr.flatten(),logg.flatten(),logb.flatten()])
    ## ベクトル
    vec = np.zeros((3,3))
    vec[0,:] = [1,1,1]
    vec[1,:] = N_pca1
    vec[2,:] = N_pca2 
    # # 肌色分布平面の法線 = 2つの色ベクトルの外積
    # # 平面から法線を求める式(vec(0,:) = [1 1 1]なので式には考慮せず)
    housen = [
        vec[1, 1] * vec[2, 2] - vec[1, 2] * vec[2, 1],
        vec[1, 2] * vec[2, 0] - vec[1, 0] * vec[2, 2],
        vec[1, 0] * vec[2, 1] - vec[1, 1] * vec[2, 0],
    ]
    # # 照明ムラ方向(陰影）と平行な成分をとおる直線と肌色分布平面との交点を求める
    # # housen：肌色分布平面の法線
    # # S：濃度空間でのRGB
    # # vec：独立成分ベクトル
    t = -(housen[0] * Skin[0] + housen[1] * Skin[1] + housen[2] * Skin[2]) / (
        housen[0] * 1 + housen[1] * 1 + housen[2] * 1
    )
    # # 陰影除去
    # # skin_flat：陰影除去したメラヘモベクトルの平面 
    Skin_flat = t+Skin
    # # ===============================================================================
    # # 肌色分布平面上のデータを，主成分分析により白色化する．
    # # ===============================================================================
    # #
    # # ゼロ平均計算用
    Skin_mean = Skin_flat.mean(axis=1)
    Skin_MeanMat = np.kron(Skin_mean[:,np.newaxis], np.ones((1, height * width)))

    # # 濃度ベクトルSの固有値と固有ベクトルを計算
    Covariance= np.cov(Skin_flat) 
    
    eig,eigvec = np.linalg.eig(Covariance)
    idx = np.argsort(eig)[::-1]
    eigvec_T = eigvec.T 
    eigvec_sorted =eigvec_T[idx]

    ## 第一主成分,第二主成分を格納する行列
    P1P2_vec = eigvec_sorted[0:2,:]

    # # 第1主成分，第2主成分
    Pcomponent = P1P2_vec @ (Skin_flat - Skin_MeanMat) ##平均を引いて正規化ベクトルかけて白色化
    
    # # ===============================================================================
    # # 独立成分分析を実行する
    # # ===============================================================================
    
    Pstd = np.std(Pcomponent, axis=1,ddof=1)
    Normalization_Matrix= np.diag(1 / Pstd)
    sensor = Normalization_Matrix @ Pcomponent  
    
    # # ===============================================================================
    # #  Burel の独立評価関数が最小となる信号を求める
    # # ===============================================================================
    np.random.seed(seed=5)
    while True:
        # Burelの独立評価値をNelder-Mead法で最小化
        # 観測値は相関がある　＝＞　独立なベクトルを求める
        # s   乱数
        # cost: 独立評価値
        s = np.zeros((1,2))
        s[0,0] = np.random.rand(1) * np.pi 
        s[0,1] = np.random.rand(1) * np.pi 
        s_solved = fmin(lambda s: f_burel(s, sensor), s)  
        cost = f_burel(s_solved,sensor)
        ## 角度から座標への変換
        x1 = np.cos(s_solved[0])
        y1 = np.sin(s_solved[0])
        x2 = np.cos(s_solved[1])
        y2 = np.sin(s_solved[1])
        H = np.array([[x1, y1], [x2, y2]])
    # # ===============================================================================
    # #  メラニン・ヘモグロビンベクトルの導出
    # # ===============================================================================
        unit1 = np.array([1.0, 0.0]).T
        unit2 = np.array([0.0, 1.0]).T

        TM = H @ Normalization_Matrix @ P1P2_vec
        
        InvTM = np.linalg.pinv(TM)
        
        c_1 = InvTM @ unit1
        c_2 = InvTM @ unit2
        
        
        ## 例外処理
        for i in range(3):
            if c_1[i] < 0:
                c_1[i] *= -1
            if c_2[i] < 0:
                c_2[i] *= -1
        
        
        ## メラニンとヘモグロビンのベクトルの関係性からどっちベクトルか推定
        c_1_norm = c_1 / ((np.sum(c_1**2)) ** (0.5))
        c_2_norm = c_2 / ((np.sum(c_2**2)) ** (0.5))
        
        if c_1_norm[1] < c_1_norm[2]:
            melanin = c_1_norm
            hemoglobin = c_2_norm
        else:
            melanin = c_2_norm
            hemoglobin = c_1_norm

        if np.all(melanin > 0) and np.all(hemoglobin > 0):
            break
        else:
            print("エラー：色ベクトルが負の値です．\n")
        
    ## -------------------------------------------------------------------------
    
    color_vector = np.zeros((2,3))
    color_vector[0] = hemoglobin
    color_vector[1] = melanin
    
    # print(f"{cost =}")
    # print(f"{melanin=}")
    # print(f"{hemoglobin=}")
    # print("color vector estimation is Done.")
    
    return melanin,hemoglobin