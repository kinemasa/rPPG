from tqdm import tqdm
import os
import cv2
import numpy as np
import sys

# utils.py までのパスを通す（相対的に）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

    
from mycommon.image_file import read_cr2_image,get_image_rotation_info,save_pigment_images
from mycommon.create_face_mask import create_face_mask


def makeSkinSeparation(INPUT_DIR,input_image_list,OUTPUT_DIR,vector):
    """"
    入力画像をメラニン・ヘモグロビン・陰影画像に分離する関数
    入力：input_image_list 画像のリスト
        OUTPUT_DIR: 出力先フォルダへのパス
        vector :メラニン・ヘモグロビンベクトル
    出力　色素成分分離後画像
    """
    
    melanin =vector[0]
    hemoglobin=vector[1]
    shading=vector[2]
    
    print('\n==== Start ====')
    print(input_image_list)
    for input_image_path in tqdm(input_image_list,desc="Processing Images",unit="image"):

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
        gamma = 1
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

        # 画像の補正
        
        skin = image_rgb.transpose(2,0,1).astype(np.float32)
        for i in range(3):
            linearSkin[i] = np.power(((skin[i]-cc)/aa), (1/gamma)-bb)/gg[i]/255

        # 顔の領域を含むマスクを作成
        img_mask = create_face_mask(image_rgb).astype(np.float32)
        img_mask = np.repeat(img_mask[np.newaxis, :, :], 3, axis=0)  # 3チャンネルに拡張
        img_mask2 = (1 / 255) + np.zeros_like(img_mask, dtype=np.float32)


        # 濃度空間 (log空間) へ
        S = -np.log(linearSkin + img_mask2) * img_mask

        # 肌色空間の起点を 0 へ
        # 必要に応じて調整するパラメーター
        MinSkin = [0, 0, 0]
        for i in range(3):
            S[i] = S[i] - MinSkin[i]
        
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

        #===============================================================================
        # 色素成分分離画像の出力
        #===============================================================================
        # 0：メラニン成分 1：ヘモグロビン成分 2：陰影成分
        L_Mel, L_Hem, L_Sha = Comp
        L_Obj = np.zeros_like(Comp, dtype=np.float32)
        
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
            img_exp = np.clip(np.exp(-img), 0, 1) * img_mask.transpose(1,2,0)
            
            # マスク外を灰色に設定
            gray_value = 192 / 255.0 
            img_exp[img_mask.transpose(1, 2, 0) == 0] = gray_value
            
            min_v = img_exp[img_mask.transpose(1,2,0)>0.0].min()
            ef_img = img_exp -min_v
            max_v = ef_img[img_mask.transpose(1,2,0)>0.0].max()
            ef_img = (ef_img / max_v)
            save_pigment_images(os.path.join(OUTPUT_DIR, output_path), ef_img)