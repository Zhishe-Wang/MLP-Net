import time

import numpy as np
import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from utils import *

import cv2
import pandas as pd

# from imageio import imsave



# from model_net.model1 import UMLP
# from model_net.model1womilf import  UMLP
# from model_net.model1noatten import UMLP
# from model_net.SMLP import StripMLPNet
# from model_net.nbmlpbest import StripMLPNet
# from model_net.nbMLPwomlpa import StripMLPNet
# from model_net.model_DNANet import Res_CBAM_block
# from model_net.parse_args_test import parse_args
# from model_net.load_param_data import load_param
# from model_net.model_DNANet import DNANet
# from uiunet import UIUNET
# from model_net.SCTransNet import SCTransNet as SCTransNet
# from model_net.MTU_Net import res_UNet, Res_block
# from model_net.nbMLPwomlpa import StripMLPNet
# from model_net.nbMLPparecnn import StripMLPNet
# from model_net.nbMLPparemlp import StripMLPNet
# from model_net.nbMLPwofuse import StripMLPNet
# from model_net.nbMLPcx import StripMLPNet
# from model_net.nbMLPnointer import StripMLPNet
# from model_net.nbMLPmixer1 import StripMLPNet
# from model_net.nbMLPmixer1 import StripMLPNet
# from model_net.kanmix import UKAN
# from model_net.nbMLPparecnn import StripMLPNet
# from model_net.nbMLPspare import StripMLPNet
# from model_net.nbmlpbest import StripMLPNet
from model_net.MLPnet import MLPNet
from utils import *

#
# from model_net.model_DNANet import DNANet
# from model_net.model_DNANet import Res_CBAM_block
# from model_net.parse_args_test import parse_args
# from model_net.load_param_data import load_param


# def make_floor(path1, path2):
#     path = os.path.join(path1, path2)
#     if os.path.exists(path) is False:
#         os.makedirs(path)
#     return path
# def save_feat(index,C,ir_atten_feat,vi_atten_feat,result_path):
#     ir_atten_feat = ir_atten_feat
#     vi_atten_feat = vi_atten_feat
#
#     ir_feat_path = make_floor(result_path, "ir_feat")
#     index_irfeat_path = make_floor(ir_feat_path, str(index))
#
#     vi_feat_path = make_floor(result_path, "vi_feat")
#     index_vifeat_path = make_floor(vi_feat_path, str(index))
#
#     for c in range(C):
#         ir_temp = ir_atten_feat[:, c, :, :].squeeze()
#         vi_temp = vi_atten_feat[:, c, :, :].squeeze()
#
#         feat_ir = ir_temp.cpu().clamp(0, 30).data.numpy()*255
#         feat_vi = vi_temp.cpu().clamp(0, 30).data.numpy()*255
#
#         ir_feat_filenames = 'ir_feat_C' + str(c) + '.png'
#         ir_atten_path = index_irfeat_path + '/' + ir_feat_filenames
#         cv2.imwrite(ir_atten_path, feat_ir)
#
#         vi_feat_filenames = 'vi_feat_C' + str(c) + '.png'
#         vi_atten_path = index_vifeat_path + '/' + vi_feat_filenames
#         cv2.imwrite(vi_atten_path, feat_vi)














def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    
    cv2.imwrite(save_path,predict_save * 255)
    
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()


    output = model(input_img.cuda())
    # input_img = input_img.cuda()   #dna net


    # if args.deep_supervision == 'True':
    #  output = model(input_img)
    #  output = output[-1]   #DNANET


    # else:
    #      output = model(input_img)
    # print(type(output))
    output = output[0]   #UIU
    # save_feat(80, 1, output, output, "C:/Users/image fusion/Desktop/wangchunfa/UMLP/savefeat/")



    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))    #0.5


    predict_save = pred_class[0].cpu().data.numpy()




    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))


    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_pred'+'.png')
    return dice_pred_tmp, iou_tmp


def overlay_and_save(original_folder, prediction_folder, output_folder, alpha=0.5):

    os.makedirs(output_folder, exist_ok=True)


    original_files = os.listdir(original_folder)
    original_files.sort(key=lambda x: int(x[:-4]))
    prediction_files = os.listdir(prediction_folder)
    prediction_files.sort(key=lambda x: int(x[:-9]))


    # 确保原始文件夹和预测文件夹中的文件数量相同
    assert len(original_files) == len(
        prediction_files), "The number of files in the original folder and the prediction folder should be the same."

    # 循环迭代并叠加图像
    for original_file, prediction_file in zip(original_files, prediction_files):
        # 构建文件路径
        original_image_path = os.path.join(original_folder, original_file)
        prediction_image_path = os.path.join(prediction_folder, prediction_file)
        output_image_path = os.path.join(output_folder, original_file)

        # 加载原始图像和分割预测图像
        original_image = cv2.imread(original_image_path)
        prediction_image = cv2.imread(prediction_image_path)

        prediction_image = cv2.resize(prediction_image, (original_image.shape[1], original_image.shape[0]))
        overlay = cv2.addWeighted(original_image, alpha, prediction_image, 1 - alpha, 0)
        cv2.imwrite(output_image_path, overlay)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    

    folder_path = config.test_dataset + "/labelcol"  # 文件夹路径
    test_num = len(os.listdir(folder_path))
    print(test_num)



    model_type = config.model_name
    model_path = "C:/Users/image fusion/Desktop/wangchunfa/UMLP/200_save_data/201_models_save/"+model_type+"/"+test_session+"/best_model-"+model_type+".pth.tar"
    # model_path = "C:/Users/image fusion/Desktop/licairong/CSN/200_save_data/201_models_save/" + model_type + "/pretrain_DNANet_model.tar"
    # model_path = "C:/Users/image fusion/Desktop/wangchunfa/UMLP/200_save_data/201_models_save/DNANet/best_model-DNANet.pth.tar"

    


    vis_path = "C:/Users/image fusion/Desktop/wangchunfa/UMLP/200_save_data/202_test_save/" + test_session + '/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    if not os.path.isdir(vis_path + 'gt/'):
        os.makedirs(vis_path + 'gt/')

    if not os.path.isdir(vis_path + 'pred/'):
        os.makedirs(vis_path + 'pred/')

    if not os.path.isdir(vis_path + 'vis/'):
        os.makedirs(vis_path + 'vis/')

    if not os.path.isdir(vis_path + 'img/'):
        os.makedirs(vis_path + 'img/')

    checkpoint = torch.load(model_path, map_location='cuda')

    # checkpoint = torch.load('result/' + args.model_dir)


    # model = UMLP(size=config.img_size, n_channels=config.n_channels, n_classes=config.n_labels, in_channels=64)
    # model = StripMLPNet(img_size=config.img_size, patch_size=4, in_chans=config.n_channels, num_classes=config.n_labels,
    #                     embed_dim=64, layers=[1, 1, 1, 1], drop_rate=0.5, norm_layer=nn.BatchNorm2d, alpha=3, use_dropout=False, patch_norm=True)
    # model = UKAN(num_classes=1, input_channels=3, deep_supervision=False, img_size=256, patch_size=8, in_chans=3,
    #              embed_dims=[16, 32, 64, 128, 256], no_kan=False,
    #              drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1])
    # args = parse_args()
    # nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
    # model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=num_blocks,
    #                nb_filter=nb_filter, deep_supervision=args.deep_supervision)
    # model = UIUNET(3, 1)

    model = MLPNet(img_size=config.img_size, patch_size=4, in_chans=config.n_channels, num_classes=config.n_labels,
                   embed_dim=64, layers=[1, 1, 1, 1], drop_rate=0.5,
                   norm_layer=nn.BatchNorm2d, alpha=3, use_dropout=False, patch_norm=True)

    #
    # config_vit = config.get_SCTrans_config()
    # model = SCTransNet(config_vit, mode='train', deepsuper=False)


    # num_blocks = [2, 2, 2, 2]
    # nb_filter = [16, 32, 64, 128, 256]
    #
    # model = res_UNet(num_classes=1, input_channels=3, block=Res_block, num_blocks=num_blocks,
    #                  nb_filter=nb_filter)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = model.cuda()
    # model.apply(weights_init_xavier)

    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    start = time.time()


    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:

        for i, (sampled_batch, names) in enumerate(test_loader, 1):

            test_data, test_label = sampled_batch['image'], sampled_batch['label']

            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255



            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)



            plt.savefig(vis_path+'gt/'+str(i)+"_lab.png", dpi=300)
            plt.close()



            # 将数据类型转换为torch.float32
            input_img = test_data.to(torch.float32)
            imgs = input_img.data.numpy()
            # print(imgs.shape)
            imgs = np.squeeze(imgs)
            imgs = np.transpose(imgs, (1, 2, 0)) * 255   #sctrans去掉  然后load数据哪里变单通道
            # print(imgs.shape)
            # imgs  = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
            img_output = vis_path + 'img/' + str(i) + '.png'
            cv2.imwrite(img_output, imgs)



            dice_pred_t,iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+'pred/'+str(i),
                                               dice_pred=dice_pred, dice_ens=dice_ens)    #保存预测图




            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()


            
    end = time.time()
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    print (end-start)

    data = {'dice_pred': [dice_pred/test_num,],
            'iou_pred': [iou_pred/test_num,]}




    # 创建DataFrame
    df = pd.DataFrame(data)

    with pd.ExcelWriter(
            'C:/Users/image fusion/Desktop/wangchunfa/UMLP/200_save_data/202_test_save/' + test_session + '/data.xlsx',
            engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)



    original_folder = vis_path+'img/'
    prediction_folder = vis_path + 'pred/'
    output_folder = vis_path + 'vis/'

    overlay_and_save(original_folder, prediction_folder, output_folder)
