import math
import torch
from core import trainer
from utils import preprocess
from data.CIKM.cikm_radar import *
from core.layers.cloud_shift import *

def wrapper_test(model, padding_CIKM_data, unpadding_CIKM_data, test_loader, args, is_save=True):
    if not args.is_training:
        model.load(args.result_model)
    test_save_root = args.gen_frm_dir
    if not os.path.exists(test_save_root):
        os.mkdir(test_save_root)
    test_output = open("%s/result.txt" % args.gen_frm_dir, 'w')
    avg_mae = 0
    avg_mse = 0
    count = 1
    real_input_flag = np.zeros(
        (args.test_batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    # 10 = 15 - 5
    output_length = args.total_length - args.input_length
    with torch.no_grad():
        for i_batch, batch_data in enumerate(test_loader):
            # ims.shape=tuple(8, 15, 101, 101, 1)
            ims = batch_data[0].numpy()
            # label tars.shape=(8, 10, 101, 101, 1)
            tars = ims[:, -output_length:]
            # 文件夹名字
            cur_fold = batch_data[1]
            cloud_shift, cloudless_shift = find_cloud_center(ims[:, :args.input_length, :, :, 0],
                                                             args.test_batch_size, args)
            cloud_shift, cloudless_shift = torch.tensor(cloud_shift), torch.tensor(cloudless_shift)
            # ims.shape=tuple(8, 15, 128, 128, 1) 填充
            ims = padding_CIKM_data(ims, args)
            # ims.shape=tuple(8, 15, 32, 32, 16) 分割
            ims = preprocess.reshape_patch(ims, args.patch_size)
            # img_gen.shape=ndarray(8, 14, 16, 32, 32)
            img_gen = model.test(ims, real_input_flag, cloud_shift, cloudless_shift)
            # img_gen.shape=ndarray(8, 14, 32, 32, 16)
            img_gen = img_gen.transpose((0, 1, 3, 4, 2))
            # img_gen.shape=ndarray(8, 14, 128, 128, 1)
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            # img_out.shape=ndarray(8, 10, 101, 101, 1)
            img_out = unpadding_CIKM_data(img_gen[:, -output_length:])
            # Calculate MSE
            mse = np.mean(np.square(img_out - tars))
            print('--- Current MSE: %.10f ---' % mse)
            avg_mse += mse
            # Calculate MAE
            mae = np.mean(np.abs(img_out-tars))
            print('--- Current MAE: %.10f ---' % mae)
            if torch.distributed.get_rank() == 0:
                print('--- Current MSE: %.10f ---' % mse, file=test_output)
                print('--- Current MAE: %.10f ---' % mae, file=test_output)
            avg_mae += mae
            img_out[img_out < 0] = 0
            img_out[img_out > 1] = 1
            img_out = (img_out*255.0).astype(np.uint8)
            # count: image_bag / batch_size
            count = count + 1
            if is_save and torch.distributed.get_rank() == 0:
                for bat_ind in range(args.test_batch_size):
                    cur_batch_data = img_out[bat_ind, :, :, :, 0]
                    cur_sample_fold = os.path.join(test_save_root, cur_fold[bat_ind])
                    if not os.path.exists(cur_sample_fold):
                        os.mkdir(cur_sample_fold)
                    for t in range(10):
                        cur_save_path = os.path.join(cur_sample_fold, 'img_'+str(t+6)+'.png')
                        cur_img = cur_batch_data[t]
                        cv2.imwrite(cur_save_path, cur_img)
    if torch.distributed.get_rank() == 0:
        avg_mse_value = str(avg_mse / count)
        print('Finally MSE:', avg_mse_value)
        print('--- Finally MSE: %.10f ---' % float(avg_mse_value), file=test_output)

        avg_mae_value = str(avg_mae / count)
        print('Finally MAE:', avg_mae_value)
        print('--- Finally MAE: %.10f ---' % float(avg_mae_value), file=test_output)

    return mse / count

def wrapper_valid(model, padding_CIKM_data, valid_loader, args):
    mse = 0
    count = 0
    output_length = args.total_length - args.input_length
    real_input_flag = np.zeros(
        (args.valid_batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    with torch.no_grad():
        for i_batch, batch_data in enumerate(valid_loader):
            ims = batch_data.numpy()
            cloud_shift, cloudless_shift = find_cloud_center(ims[:, :args.input_length, :, :, 0],
                                                             args.valid_batch_size, args)
            cloud_shift, cloudless_shift = torch.tensor(cloud_shift), torch.tensor(cloudless_shift)
            ims = padding_CIKM_data(ims, args)
            ims = preprocess.reshape_patch(ims, args.patch_size)
            img_out = model.valid(ims, real_input_flag, cloud_shift, cloudless_shift)
            tars = torch.FloatTensor(ims[:, -output_length:])
            tars = tars.permute(0, 1, 4, 2, 3).contiguous()
            img_out = torch.FloatTensor(img_out[:, -output_length:])
            mse += torch.mean(torch.square(img_out - tars))
            count = count+1

    return mse/count

def wrapper_train(model, train_loader, valid_loader, padding_CIKM_data, schedule_sampling, args):
    # torch.distributed.init_process_group("nccl")
    if args.pretrained_model:
        model.load(args.pretrained_model)
    log_output = open("%s/log.txt" % args.save_dir, 'w')
    eta = args.sampling_start_value
    best_mse = math.inf
    tolerate = 0
    iter_num = len(train_loader.dataset) / (4 * args.train_batch_size)

    for itr in range(1, args.max_iterations + 1):
        # 分布式训练
        # 设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        train_loader.sampler.set_epoch(itr)
        for i_batch, batch_data in enumerate(train_loader):
            # ims.shape=[35, 15, 101, 101, 1]
            ims = batch_data.numpy()
            # 寻找云图第一帧和第五帧用于求解位移 (X[:,1]输出结果是：数组的第一维的第1个数据)
            cloud_shift, cloudless_shift = find_cloud_center(ims[:, :args.input_length, :, :, 0],
                                                             args.train_batch_size, args)
            cloud_shift, cloudless_shift = torch.tensor(cloud_shift), torch.tensor(cloudless_shift)
            # ims.shape=[35, 15, 128, 128, 1]
            ims = padding_CIKM_data(ims, args)
            # ims.shape=[35, 15, 32, 32, 16]
            ims = preprocess.reshape_patch(ims, args.patch_size)
            eta, real_input_flag = schedule_sampling(eta, itr, args)
            cost = trainer.train(model, ims, real_input_flag, args, cloud_shift, cloudless_shift)
            if (i_batch+1) % args.display_interval == 0:
                print('Epoch: [%d/%d], Iteration: [%d/%d], Training loss: %.20f, lr: %s' %
                      (itr, args.max_iterations, i_batch, iter_num, cost, args.lr))
                print('Epoch: [%d/%d], Iteration: [%d/%d], Training loss: %.20f, lr: %s' %
                      (itr, args.max_iterations, i_batch, iter_num, cost, args.lr), file=log_output)
        if (itr+1) % args.test_interval == 0:
            print('Validation')
            curr_mse = wrapper_valid(model, padding_CIKM_data, valid_loader, args)
            print('--- Best MSE: %.10f, Curr MSE: %.10f ---' % (best_mse, curr_mse))
            print('--- Best MSE: %.10f, Curr MSE: %.10f ---' % (best_mse, curr_mse), file=log_output)

            if curr_mse < best_mse:
                best_mse = curr_mse
                tolerate = 0
                model.save(itr)
            else:
                model.save(itr)
                tolerate = tolerate+1

            # Stop automatically when meeting the best result
            if tolerate == args.limit:
                print('The Best MSE is: ', str(best_mse))
                break

