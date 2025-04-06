# skin_separation/separation.py

import os
import numpy as np
from tqdm import tqdm
import cv2
from .image_io import read_cr2_image, save_image, get_image_rotation_info
from .create_mask import (
    create_black_mask, create_face_mask, create_hsv_mask,
    create_eye_mask
)
from .bias import bias_adjast_1d
from .visualization_graph import plot_3d_skinFlat_plotly  # 使わないなら消してOK


def makeSkinSeparation(
    INPUT_DIR,
    input_image_list,
    OUTPUT_DIR,
    vector,
    mask_type="face",
    bias_flag=False,
    bias_fixed=False,
    bias_Mel=0.0,
    bias_Hem=0.0
):
    """
    入力画像をメラニン・ヘモグロビン・陰影画像に分離して保存する関数

    Parameters:
        INPUT_DIR (str): 入力フォルダパス
        input_image_list (list): 入力画像のパス一覧
        OUTPUT_DIR (str): 出力フォルダパス
        vector (list): [melanin, hemoglobin, shading] 各RGBベクトル
        mask_type (str): マスク種別（例："face", "black", "face-eye-hsv" など）
        bias_flag (bool): バイアス補正を行うかどうか
        bias_fixed (bool): 固定バイアスか自動調整か
        bias_Mel (float): メラニンのバイアス（bias_fixed=True時）
        bias_Hem (float): ヘモグロビンのバイアス
    """

    melanin, hemoglobin, shading = vector

    print('\n==== Start Skin Separation ====')
    for input_image_path in tqdm(input_image_list, desc="Processing Images", unit="image"):
        print(input_image_path)
        image_basename = os.path.splitext(os.path.basename(input_image_path))[0]
        image_type = os.path.splitext(os.path.basename(input_image_path))[1]

        #===============================================================================
        # 画像の読み込み
        #===============================================================================
        try:
            if image_type == '.npy': # Linear RGB, HDR after AIRAW and ReAE.
                image_rgb = 0.8 * np.load(input_image_path)
                image_rgb = 255.0 * image_rgb
                image_rgb = cv2.resize(image_rgb, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC)
                image_rgb = image_rgb.clip(min=0.0)
            
            elif image_type ==".CR2":
                print("CR2")
                image_rgb = read_cr2_image(input_image_path)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # RGB -> BGRに変換
                OUTPUT_PNG_DIR = INPUT_DIR+"\\png\\"
                os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)
                cv2.imwrite(OUTPUT_PNG_DIR+"\\"+image_basename+".png",image_bgr)
            else:
                image_rgb = cv2.cvtColor(cv2.imread(input_image_path, -1), cv2.COLOR_BGR2RGB)
                rotation_info = get_image_rotation_info(input_image_path)
                if rotation_info ==6:
                    image_rgb = cv2.rotate(image_rgb,cv2.ROTATE_90_CLOCKWISE)

                
        except:
            Exception('ERROR: Input image file was not found.')

        image_height = image_rgb.shape[0]
        image_width = image_rgb.shape[1]
        

        
        #===============================================================================
        # 画像調整用パラメーターの設定
        #===============================================================================
        # γ 補正用パラメータの取得
        ##固定値
        
        aa = 1
        bb = 0
        gamma =1 
        cc = 0
        gg = [1, 1, 1]
        DC = 1 / 255
        
        # 色ベクトルと照明強度ベクトル
        vec = np.zeros((3, 3), dtype=np.float32)
        vec[0] = np.array([1.0, 1.0, 1.0])
        vec[1] = melanin
        vec[2] = hemoglobin

        #===============================================================================
        # 画像情報を濃度空間へ
        #===============================================================================

        # 配列の初期化
        linearSkin = np.zeros_like(image_rgb, dtype=np.float32).transpose(2,0,1)
        S = np.zeros_like(image_rgb, dtype=np.float32).transpose(2,0,1)

        # 画像の補正 (画像の最大値を1に正規化)
        
        skin = image_rgb.transpose(2,0,1).astype(np.float32)
        for i in range(3):
            linearSkin[i] = np.power(((skin[i]-cc)/aa), (1/gamma)-bb)/gg[i]/255

        # 顔の領域を含むマスクを作成
        if mask_type == "face":
            print("Using face mask.")
            img_mask = create_face_mask(image_rgb)
        elif mask_type == "black":
            print("Using black mask.")
            img_mask = create_black_mask(image_rgb)
        elif mask_type == "hsv":
            print("Using HSV mask.")
            # 色相、彩度、明度の範囲を指定
            hue_range = (0, 50)  
            saturation_range = (50, 255)
            value_range = (50, 255)
            img_mask = create_hsv_mask(image_rgb, hue_range, saturation_range, value_range)
        elif mask_type == "face-hsv":
            print("Using HSV mask.")
            # 色相、彩度、明度の範囲を指定
            hue_range = (0, 50)  
            saturation_range = (50, 255)
            value_range = (50, 255)
            img_mask1 = create_hsv_mask(image_rgb, hue_range, saturation_range, value_range)
            img_mask2 = create_face_mask(image_rgb)
            
            img_mask = img_mask1 * img_mask2
        elif mask_type == "black-hsv":
            print("black HSV mask.")
            # 色相、彩度、明度の範囲を指定
            hue_range = (0, 50)  
            saturation_range = (50, 255)
            value_range = (50, 255)
            img_mask1 = create_hsv_mask(image_rgb, hue_range, saturation_range, value_range)
            img_mask2 = create_black_mask(image_rgb)
            
            img_mask = img_mask1 * img_mask2
        elif mask_type == "face-eye":
            print("face-eye-hsv mask.")
            img_mask1 = create_eye_mask(image_rgb)
            img_mask2 = create_face_mask(image_rgb)
            img_mask = img_mask1 * img_mask2
            
        elif mask_type == "face-eye-hsv":
            print("face-eye-hsv mask.")
            
            hue_range = (0, 50)  
            saturation_range = (50, 255)
            value_range = (50, 255)
            img_mask1 = create_eye_mask(image_rgb)
            img_mask2 = create_face_mask(image_rgb)
            img_mask3 = create_hsv_mask(image_rgb, hue_range, saturation_range, value_range)
        
            img_mask = img_mask1 * img_mask2* img_mask3
        else:
            raise ValueError(f"Invalid mask_type: {mask_type}. Choose 'face' or 'black'.")
        
        img_mask = np.repeat(img_mask[np.newaxis, :, :], 3, axis=0)  # 3チャンネルに拡張
        img_mask2 = (1 / 255) + np.zeros_like(img_mask, dtype=np.float32)

        linearSkin = np.zeros_like(image_rgb, dtype=np.float32).transpose(2, 0, 1)
        S = np.zeros_like(image_rgb, dtype=np.float32).transpose(2, 0, 1)

        skin = image_rgb.transpose(2, 0, 1).astype(np.float32)
        for i in range(3):
            linearSkin[i] = np.power((skin[i] / 2), 1) / 255


        # 濃度空間 (log空間) へ
        S = -np.log(linearSkin + img_mask2) * img_mask

        
        #===============================================================================
        # 陰影成分の除去
        #===============================================================================
        
        # 肌色分布平面の法線 = 2つの色ベクトルの外積
        # 平面から法線を求める式 (vec(1,:) = [1 1 1] なので式には考慮せず)
        norm = [
            vec[1,1]*vec[2,2]-vec[1,2]*vec[2,1],
            vec[1,2]*vec[2,0]-vec[1,0]*vec[2,2],
            vec[1,0]*vec[2,1]-vec[1,1]*vec[2,0],
            ]

        # 照明ムラ方向と平行な成分をとおる直線と肌色分布平面との交点を求める
        # housen：肌色分布平面の法線
        # S：濃度空間でのRGB
        # vec：独立成分ベクトル
        t = -(norm[0]*S[0] + norm[1]*S[1] + norm[2]*S[2]) / (norm[0]*vec[0,0]+norm[1]*vec[0,1]+norm[2]*vec[0,2])

        # 陰影除去
        # skin_flat：陰影除去したメラヘモベクトルの平面
        # rest：陰影成分
        skin_flat = (t[np.newaxis,:,:].transpose(1,2,0)*vec[0]).transpose(2,0,1) + S
        rest = S - skin_flat
        #===============================================================================
        # 色素濃度の計算
        #===============================================================================
        # 混合行列と分離行列
        CompExtM = np.linalg.pinv(np.vstack([melanin, hemoglobin]).transpose())

        # 各画素の色素濃度の取得
        #　　濃度分布 ＝ [メラニン色素； ヘモグロビン色素]
        #　　　　　　 ＝ 肌色ベクトル (陰影除去後) × 分離行列
        Compornent = np.dot(CompExtM, skin_flat.reshape(3, image_height * image_width))
        Compornent = Compornent.reshape(2, image_height, image_width)
        Comp = np.vstack([Compornent, (rest[0])[np.newaxis,:,:]])
        # 0：メラニン成分 1：ヘモグロビン成分 2：陰影成分
        L_Mel, L_Hem, L_Sha = Comp
        L_Obj = np.zeros_like(Comp, dtype=np.float32)
        
        
        #====================================================================================
        # バイアス補正
        #=================================================================================
        #================
        ## 1次元
        #================
        if bias_flag == True:
            if bias_fixed == True:
                if bias_Mel <0:
                    L_Mel = L_Mel-bias_Mel
                if bias_Hem <0:
                    L_Hem = L_Hem-bias_Hem
            else :
                L_Mel,L_Hem = bias_adjast_1d(L_Mel,L_Hem)
                
        
        # plot_3d_skinFlat_plotly(skin_flat,melanin,hemoglobin,vector)
        #===============================================================================
        # 色素成分分離画像の出力
        #===============================================================================
        # L_Mel = np.clip(L_Mel,0,3)
        # L_Hem = np.clip(L_Hem,0,1)
        
        # どちらかが 絶対値5 を超えるピクセルを外れ値として0にする
        mask = (np.abs(L_Hem) > 5) | (np.abs(L_Mel) > 5)
        L_Hem[mask] = 0
        L_Mel[mask] = 0
        for param_index in range(3):

            # 各チャネル情報を取得する。
            if param_index == 0:
                for chn_index in range(3):
                    L_Obj[chn_index] = melanin[chn_index] * L_Mel
                    output_path = '{}_2_Mel.png'.format(image_basename)
            elif param_index == 1:
                for chn_index in range(3):
                    L_Obj[chn_index] = hemoglobin[chn_index] * L_Hem
                    output_path = '{}_3_Hem.png'.format(image_basename)
                
            else:
                for chn_index in range(3):
                    L_Obj[chn_index] =shading[chn_index] * L_Sha
                    output_path = '{}_1_Sha.png'.format(image_basename)

            # 可視化して保存する。最大値と最小値で規格化する
            img = L_Obj.transpose(1,2,0)
            
            img_exp = np.exp(-img) * img_mask.transpose(1,2,0)
            
            # マスク外を灰色に設定
            gray_value = 192 / 255.0 
            img_exp[img_mask.transpose(1, 2, 0) == 0] = gray_value
            
            ef_img =img_exp
            save_image(os.path.join(OUTPUT_DIR, output_path), ef_img)



    print("Skin separation complete.")