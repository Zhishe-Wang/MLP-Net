

""" 超参数 """

save_model = True
tensorboard = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)


model_name = 'MLPNet'

cosineLR = True
learning_rate = 1e-3 #-3
batch_size = 8

n_channels = 3
n_labels = 1000
epochs = 600
img_size = 256
print_frequency = 1
save_frequency = 500
vis_frequency = 200
early_stopping_patience = 200


test_session = "Test_session_10.22_12h13"  #MLP NET


# """ 路径 """
train_dataset = 'C:/Users/image fusion/Desktop/wangchunfa/UMLP/100_datasets_1/110_训练集'
val_dataset = 'C:/Users/image fusion/Desktop/wangchunfa/UMLP/100_datasets_1/120_验证集'
test_dataset = 'C:/Users/image fusion/Desktop/wangchunfa/UMLP/100_datasets_1/130_测试集'  #irstd 1k


session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = 'C:/Users/image fusion/Desktop/wangchunfa/UMLP/200_save_data/201_models_save/'+ model_name +'/' + session_name +'/'
model_path         = save_path
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'




