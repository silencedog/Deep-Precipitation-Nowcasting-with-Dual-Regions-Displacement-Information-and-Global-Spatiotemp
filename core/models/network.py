import torch
import torch.nn as nn
from core.layers.GLSTM import LSTMCell
# Kalman
from core.layers.cloud_shift import CloudShift

class Network(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(Network, self).__init__()
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_ch = configs.img_channel * (configs.patch_size ** 2)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.neighbour = 3
        self.motion_hidden = 2 * self.neighbour * self.neighbour


        cell_list = []
        for i in range(num_layers):
            in_channel = self.patch_ch if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.patch_height, self.patch_width, configs.filter_size, configs.stride, configs.layer_norm)
            )
        enc_list = []
        for i in range(num_layers - 1):
            enc_list.append(
                nn.Conv2d(num_hidden[i], num_hidden[i] // 4, kernel_size=configs.filter_size, stride=2,
                          padding=configs.filter_size // 2),
            )
        dec_list = []
        for i in range(num_layers - 1):
            dec_list.append(
                nn.ConvTranspose2d(num_hidden[i] // 4, num_hidden[i], kernel_size=4, stride=2,
                                   padding=1),
            )
        gate_list = []
        for i in range(num_layers - 1):
            gate_list.append(
                nn.Conv2d(num_hidden[i] * 2, num_hidden[i], kernel_size=configs.filter_size, stride=1,
                          padding=configs.filter_size // 2),
            )

        self.gate_list = nn.ModuleList(gate_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch, 1, stride=1, padding=0, bias=False)
        self.conv_first_v = nn.Conv2d(self.patch_ch, num_hidden[0], 1, stride=1, padding=0, bias=False)
        self.CloudShift = CloudShift()

    def forward(self, all_frames, mask_true, batch_size, cloud_shift, cloudless_shift):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = all_frames
        mask_true = mask_true
        dis_frames = []
        next_frames = []
        final_frames = []
        h_t = []
        c_t = []
        h_t_conv = []
        h_t_conv_offset = []
        mean = []

        for i in range(self.num_layers):
            zeros = torch.empty(
                [batch_size, self.num_hidden[i], self.patch_height, self.patch_width])
            nn.init.xavier_normal_(zeros)
            h_t.append(zeros)
            c_t.append(zeros)

        for i in range(self.num_layers - 1):
            zeros = torch.empty(
                [batch_size, self.num_hidden[i] // 4, self.patch_height // 2,
                 self.patch_width // 2])
            nn.init.xavier_normal_(zeros)
            h_t_conv.append(zeros)
            zeros = torch.empty(
                [batch_size, self.motion_hidden, self.patch_height // 2, self.patch_width // 2])
            nn.init.xavier_normal_(zeros)
            h_t_conv_offset.append(zeros)
            mean.append(zeros)

        mem = torch.empty([batch_size, self.num_hidden[0], self.patch_height, self.patch_width])
        motion_highway = torch.empty(
            [batch_size, self.num_hidden[0], self.patch_height, self.patch_width])
        # rainfll motion
        CloudShift = torch.empty(
            [batch_size, self.num_hidden[0], self.patch_height, self.patch_width])
        # non-rainfll motion
        CloudlessShift = torch.empty(
            [batch_size, self.num_hidden[0], self.patch_height, self.patch_width])
        nn.init.xavier_normal_(mem)
        nn.init.xavier_normal_(motion_highway)
        nn.init.xavier_normal_(CloudShift)
        nn.init.xavier_normal_(CloudlessShift)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            motion_highway = self.conv_first_v(net)
            h_t[0] = h_t[0].cuda() + CloudlessShift.cuda()
            h_t[0] = CloudShift.cuda() + h_t[0].cuda()
            h_t[0], c_t[0], mem, motion_highway = self.cell_list[0](net, h_t[0], c_t[0], mem, motion_highway)
            net = self.enc_list[0](h_t[0])
            h_t_tmp = self.dec_list[0](net)
            o_t = torch.sigmoid(self.gate_list[0](torch.cat([h_t_tmp, h_t[0]], dim=1)))
            h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

            for i in range(1, self.num_layers - 1):
               
                h_t[i] = h_t[i].cuda() + CloudlessShift.cuda()
                h_t[i] = h_t[i].cuda()  + CloudShift.cuda()
                h_t[i], c_t[i], mem, motion_highway = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], mem, motion_highway)
                net = self.enc_list[i](h_t[i])
                h_t_tmp = self.dec_list[i](net)
                o_t = torch.sigmoid(self.gate_list[i](torch.cat([h_t_tmp, h_t[i]], dim=1)))
                h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]

            h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway = self.cell_list[
                self.num_layers - 1](
                h_t[self.num_layers - 2], h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway)

            if (t-1) > -1:
                cloud_trend = h_t[self.num_layers - 1] - dis_frames[t-1]
                # piexl_value -> norm: 5 dbz ~ 0
                generate_cloud = cloud_trend.clamp(min=0)
                disapper_cloud = cloud_trend.clamp(max=0)
                CloudShift, CloudlessShift = self.CloudShift(generate_cloud.cuda(), disapper_cloud.cuda(), cloud_shift,
                                                             cloudless_shift)

            dis_frames.append(h_t[self.num_layers - 1])
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        final_frames.append(next_frames)

        return final_frames