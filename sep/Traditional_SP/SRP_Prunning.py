import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
from scipy.fftpack import fftshift
from tqdm import tqdm
import pyroomacoustics as pra
import matplotlib
from .Patch_3D import Patch
import time
import math
import torch
import os 
import pickle
from .MUSIC_block import MUSIC
from .TOPS_block import TOPS

err_tolerance = 0.2

def hyperbola_offset(offset, pos, sample_offsets, width): 
    '''
        check whether offset vector is within the Pacth(sample_offset, width) 
    '''
    # z = np.sum(np.abs(offset - sample_offsets) > width/2, axis = -1)
    # return pos[z==0]
    z = 1
    for i in range(offset.shape[-1]):
        z = z & (offset[..., i] >= sample_offsets[i] - width/2) & (offset[..., i] <= sample_offsets[i] + width/2)
    return pos[z==1]

def hyperbola_area_sample(sample_list, sample_offsets, width):
    '''
        check whether offset vector is within the Pacth(sample_offset, width) 
    '''
    z = 1
    for i in range(sample_offsets.shape[-1]):
        bound1 = [sample_offsets[i] - width/2, sample_offsets[i] + width/2 ]
        z = z & (sample_list[:, i] >= bound1[0]) & (sample_list[:, i] <= bound1[1])

    return z.astype(int) 

def hyperbola_area_init(Axis_range, sample_offsets, width, Pos5, Offset5, Pos1, Offset1):
    '''
        input the patch (sample_offsets, width) and return the 3D spaces
    '''
    with_points = hyperbola_offset(Offset5, Pos5, sample_offsets, width)
    if with_points.shape[0] == 0:
        return None
    else:
        X_min, X_max = with_points[:, 0].min() - 0.05, with_points[:, 0].max() + 0.05
        X_min = max([Axis_range[0][0], X_min])
        X_max = min([Axis_range[0][1], X_max])
        Xindex_min, Xindex_max= int(np.floor((X_min - Axis_range[0][0])/0.01)), int(np.ceil((X_max - Axis_range[0][0])/0.01))
        Y_min, Y_max = with_points[:, 1].min() - 0.05, with_points[:, 1].max() + 0.05
        Y_min = max([Axis_range[1][0], Y_min])
        Y_max = min([Axis_range[1][1], Y_max])
        
        Yindex_min, Yindex_max= int(np.floor((Y_min - Axis_range[1][0])/0.01)), int(np.ceil((Y_max - Axis_range[1][0])/0.01))
        Pos1_cut = Pos1[Yindex_min:Yindex_max, Xindex_min:Xindex_max, :, :]
        Offset1_cut = Offset1[Yindex_min:Yindex_max, Xindex_min:Xindex_max, :, :]
        with_points = hyperbola_offset(Offset1_cut, Pos1_cut, sample_offsets, width)
    return with_points.T



'''
    Grid cluster class  
'''
class Grid_cluster(object):
    def __init__(self, sample_offset, pos, idx):
        self.sample_offset = sample_offset
        self.grids = pos
        self.index = idx

    def compare_offset(self, offset2, sample_resolution):
        delta = np.abs(self.sample_offset - offset2).max()
        if delta <= sample_resolution/2:
            return True
        else:
            return False
    def equal(self, offset2):
        return np.array_equal(self.sample_offset, offset2)

    def add_grid(self, pos, idx):
        self.grids.append(pos)
        self.index.append(idx)
    
    def cluster_size(self):
        return len(self.grids)

    def center_pos(self):
        return np.mean(self.grids, axis = 0)


    def dump(self):
        return [self.sample_offset, self.grids, self.index]


'''
    Main module for SRP-PHAT Prunning process  
'''
class SRP_PHAT(object):
    def __init__(
        self,
        mic_pos, 
        freq_bins,
        Range_spk,
        C = 343,
        FS = 16000,
        n_fft = 1024,
        grid_size = 0.06,
        grid_size_z = 0.1,
        sample_resolution = 4,
        threshold = 0.03,
        WIDTH = 8,
        device = None,
        cached = False,
        cached_name = None
    ):
        self.device = device ### device for SRP-PHAT    
        self.C = C   ### sound speed
        self.FS = FS ## sampling offset
        self.freq_bins = freq_bins ### bandwidth to apply SRP-PHAT
        self.n_fft = n_fft ### FFT length for SRP-PHAT

        self.mic_pos= mic_pos ## microphone positions
        self.num_mic = mic_pos.shape[0] ## microphone number
        self.mic_center = mic_pos.mean(0)  ## microphone array center
        self.sample_resolution = sample_resolution ### sample resolution to divide the TDoA space
        self.WIDTH = WIDTH ### width of inital patch


        self.Range_spk = Range_spk ### the range of interest of speakers
        self.x_grids = np.arange(Range_spk[0], Range_spk[1], grid_size)   
        self.Lx = self.x_grids.shape[0]
        self.y_grids = np.arange(Range_spk[2], Range_spk[3], grid_size)
        self.Ly = self.y_grids.shape[0]
        self.z_grids = np.arange(Range_spk[4], Range_spk[5], grid_size_z)
        self.Lz = self.z_grids.shape[0]    
        
        self.dis_matrix = np.zeros(( self.Lx ,  self.Ly))
        for ix in range(0, self.Lx ):
            for iy in range(0, self.Ly ):
                candiate_pos = np.array([self.x_grids[ix],  self.y_grids[iy]])
                self.dis_matrix[ix][iy] = np.linalg.norm(candiate_pos - self.mic_center[:2]) + 1e-8

        self.Axis_range = [[Range_spk[0], Range_spk[1]], [Range_spk[2], Range_spk[3]], [Range_spk[4], Range_spk[5]] ]        

        ### initialize the 0.05 cm size grids and sample offset
        xx = np.arange(Range_spk[0], Range_spk[1], 0.05)
        yy = np.arange(Range_spk[2], Range_spk[3], 0.05)
        zz = np.arange(Range_spk[4], Range_spk[5], 0.1)
        X, Y, Z = np.meshgrid(xx, yy, zz)
        self.Pos_5 = np.stack((X,Y,Z), axis = 3)
        

        ### initialize the 0.01 cm size grids and sample offset
        xx = np.arange(Range_spk[0], Range_spk[1], 0.01)
        yy = np.arange(Range_spk[2], Range_spk[3], 0.01)
        zz = np.arange(Range_spk[4], Range_spk[5], 0.1)
        X, Y, Z = np.meshgrid(xx, yy, zz)
        self.Pos_1 = np.stack((X,Y,Z), axis = 3)
        self.Offset_5 = [] 
        self.Offset_1 = []
        for i in range(1, mic_pos.shape[0]):
            offs = np.linalg.norm(self.Pos_5 - mic_pos[i, :], axis = 3)  / C * FS  - np.linalg.norm(self.Pos_5 - mic_pos[0, :], axis = 3) / C * FS
            self.Offset_5.append(offs)
            offs = np.linalg.norm(self.Pos_1 - mic_pos[i, :], axis = 3)  / C * FS  - np.linalg.norm(self.Pos_1 - mic_pos[0, :], axis = 3) / C * FS
            self.Offset_1.append(offs)
        self.Offset_5 = np.stack(self.Offset_5, axis = 3)
        self.Offset_1 = np.stack(self.Offset_1, axis = 3)


        #### calculate the keepout range from the desk to the speakers positions
        KEEPOUT = 0.2
        self.threshold = threshold
        mic_array_minx = np.min(mic_pos[:, 0]) - KEEPOUT
        mic_array_miny = np.min(mic_pos[:, 1]) - KEEPOUT
        mic_array_maxx = np.max(mic_pos[:, 0]) + KEEPOUT
        mic_array_maxy = np.max(mic_pos[:, 1]) + KEEPOUT
        self.array_border= [mic_array_minx, mic_array_miny, mic_array_maxx, mic_array_maxy]
        self.debug_indexs = []  

        need_initialize = True
        if cached and cached_name is not None:
            clusters_name = os.path.join(cached_name, "init_cached.pkl")
            if os.path.exists(clusters_name): # os.path.exists(npz_name) and
                print("Loda pickle files!")
                with open(clusters_name, "rb") as file:
                    cached_data = pickle.load(file)
                file.close()
                self.POWER_MAP = cached_data["POWER_MAP"]
                self.POWER_INDEX = cached_data["POWER_INDEX"]
                self.grids = cached_data["grids"]
                self.clusters = [Grid_cluster(*attributes) for attributes in cached_data["cluster"]]
                need_initialize = False

        ### calculate the mapping tabel between the 3D space and TDoA space
        if need_initialize:
            self.sample_tree = np.zeros((self.Lx , self.Ly, self.Lz, self.num_mic), dtype = int)
            self.POWER_MAP = np.zeros((self.Lx , self.Ly, self.Lz)) ### save
            self.POWER_INDEX = np.zeros((self.Lx , self.Ly, self.Lz), dtype = int) ### save
            self.grids = [] ### save
            self.clusters = [] ### save
            self.Map_3D_TDoA()

            if cached and cached_name is not None:
                cached_data = {}
                cached_data["POWER_MAP"] = self.POWER_MAP
                cached_data["POWER_INDEX"] = self.POWER_INDEX
                cached_data["grids"] = self.grids
                cached_data["cluster"] = [i.dump() for i in self.clusters]

                clusters_name = os.path.join(cached_name, "init_cached.pkl")
                with open(clusters_name, "wb") as file:
                    pickle.dump(cached_data, file)
                print("Sucessfully save init_cached.pkl")
                file.close()



        ### calculate the mod vector for the SRP-PHAT
        self.mode_vec = self.generate_mod_vector(FS, n_fft) ## save
        ar = np.arange(self.num_mic)
        av = ar[:, None]
        # stack of cov. matrices
        mask_triu = (av < av.T).flatten()
        middle_vec = np.moveaxis(self.mode_vec, 2, 0)
        multiple_vec = np.matmul(middle_vec[:, :, :, None], np.conj(middle_vec[:, :, None, :]))
        multiple_vec = multiple_vec.reshape((multiple_vec.shape[0], multiple_vec.shape[1], self.num_mic * self.num_mic))[:, :, mask_triu]
        self.multiple_vec = multiple_vec  ## save

        self.SRP_map = torch.zeros(self.grids.shape[0]) ### no save
        if device is not None:
            self.SRP_map = self.SRP_map.to(device)

        mode_mat_flat= self.multiple_vec
        mode_mat_flat_th = torch.tensor(mode_mat_flat)
        
        if self.device is not None:
            mode_mat_flat_th = mode_mat_flat_th.to(self.device)
        
        self.mode_mat_flat_real = torch.real(mode_mat_flat_th) ## save
        self.mode_mat_flat_imag = torch.imag(mode_mat_flat_th) ## save
        
        self.peak_high_prio = []
        self.peak_low_prio = []

    def reset(self):
        self.peak_high_prio = []
        self.peak_low_prio = []


        self.SRP_map = torch.zeros(self.grids.shape[0])
        if self.device is not None:
            self.SRP_map = self.SRP_map.to(self.device)

    def calculate_offset_pair(self, pos):
        offsets = []
        for i in range(1, self.num_mic):
            delta_dis = np.linalg.norm(pos - self.mic_pos[i]) - np.linalg.norm(pos - self.mic_pos[0])
            delta = delta_dis/self.C * self.FS
            offsets.append(delta)
        return np.array(offsets)


    def check_valid(self, idx):
        if idx[0] < 0 or idx[0] >= self.Lx or idx[1] < 0  or idx[1] >= self.Ly or idx[2] < 0 or idx[2] >= self.Lz:
            return False
        else:
            x = self.x_grids[idx[0]]
            y = self.y_grids[idx[1]]
            if ( x > self.array_border[0] and y > self.array_border[1] and x < self.array_border[2] and y < self.array_border[3]):
                return False
            else:
                return True

    def search_cluster(self, idx):
        if self.sample_tree[idx[0], idx[1], idx[2], 0] == 0:
            raise ValueError("invalid start point!!")
        ix, iy, iz = idx
        self.sample_tree[ix, iy, iz, 0] = 0

        list_to_check = [idx]
        list_to_merge = []

        visited_map = np.zeros((self.Lx , self.Ly, self.Lz), dtype = int)
        visited_map[ix, iy, iz] = 1
        now_offset = self.sample_tree[ix, iy, iz, 1:]
        pos = [self.x_grids[ix], self.y_grids[iy], self.z_grids[iz]]
        new_cluster = Grid_cluster(now_offset, [pos], [idx])
        while True:
            if len(list_to_check) <= 0:
                break
            ix2, iy2, iz2 = list_to_check.pop(0)

            for new_x in range(ix2-1, ix2+2):
                for new_y in range(iy2-1, iy2+2):
                    for new_z in range(iz2-1, iz2+2):
                        if not self.check_valid([new_x, new_y, new_z]): continue
                        if self.sample_tree[new_x, new_y, new_z, 0] == 0 or visited_map[new_x, new_y, new_z]: continue
                        
                        visited_map[new_x, new_y, new_z] = 1
                        new_offset = self.sample_tree[new_x, new_y, new_z, 1:]
                        if np.array_equal(now_offset, new_offset):
                            list_to_merge.append([new_x, new_y, new_z])
                            list_to_check.append([new_x, new_y, new_z])
        for ix, iy, iz in list_to_merge: 
            
            pos = [self.x_grids[ix], self.y_grids[iy], self.z_grids[iz]]
            new_cluster.add_grid(pos, [ix, iy, iz])
            self.sample_tree[ix, iy, iz, 0] = 0
    
        return new_cluster

    def Map_3D_TDoA(self):
        for ix in range(self.Lx ):
            for iy in range(self.Ly):
                for iz in range(self.Lz):
                    x = self.x_grids[ix]
                    y = self.y_grids[iy]
                    z = self.z_grids[iz]
                    if not self.check_valid([ix, iy, iz]):
                        continue
                    pos = np.array([x, y, z])
                    offset = self.calculate_offset_pair(pos)

                    offset = offset/self.sample_resolution
                    offset = np.round(offset).astype(int)
                    offset *= self.sample_resolution
                    self.sample_tree[ix, iy, iz, 0] = 1
                    self.sample_tree[ix, iy, iz, 1:] = offset

        for ix in range(self.Lx):
            for iy in range(self.Ly):
                for iz in range(self.Lz):
                    if self.sample_tree[ix, iy, iz, 0] == 0:
                        continue
                    new_cluster = self.search_cluster([ix, iy, iz])
                    self.clusters.append(new_cluster)
                    self.grids.append(new_cluster.center_pos())
        
        self.grids = np.array(self.grids)
        self.SRP_times = len(self.clusters)
        print("MAX sportforming number: ", len(self.clusters))

        
    def fill_powermap_torch(self):
        if self.device is not None:
            SRP_MAP_CPU = self.SRP_map.cpu().numpy()
        else:
            SRP_MAP_CPU = self.SRP_map.numpy()

        for i, c in enumerate(self.clusters):
            idxes = c.index
            for ix, iy, iz in idxes:
                self.POWER_MAP[ix, iy, iz] = SRP_MAP_CPU[i]
                self.POWER_INDEX[ix, iy, iz] = i

    def fill_powermap(self):
        for i, c in enumerate(self.clusters):
            idxes = c.index
            for ix, iy, iz in idxes:
                self.POWER_MAP[ix, iy, iz] = self.SRP_map[i].item()
                self.POWER_INDEX[ix, iy, iz] = i
    


    def generate_mod_vector(self, fs, nfft):
        px = self.grids[None, None, :, 0]
        py = self.grids[None, None, :, 1]
        pz = self.grids[None, None, :, 2]
        mx = self.mic_pos[None, :, None, 0]
        my = self.mic_pos[None, :, None, 1]
        mz = np.zeros_like(mx)
        dist = np.sqrt((px - mx) ** 2 + (py - my) ** 2 +  (pz)** 2)
        dist = dist/self.C
        omega = 2 * np.pi * fs * self.freq_bins / nfft

        mode_vec = np.exp(1j * omega[:, None, None] * dist)

        return mode_vec


    def SRP_Map_WINDOW_new(self, signal, window= 36000, tol = 1e-8):
        return self.SRP_Map_WINDOW_torch(signal, window, tol)

    def SRP_Map_WINDOW_torch(self, signal, window= 36000, tol = 1e-8):
        self.MAX_POWER = -100 
        n_mic = signal.shape[0]
        assert self.mic_pos.shape[0] == n_mic
        nfft = self.n_fft
        
        step = window//2
        T = signal.shape[1]
        frame_number = T//step - 1
        ar = np.arange(n_mic)
        av = ar[:, None]
        mask_triu = (av < av.T).flatten()
        

        for j in range(0, frame_number):
            if (j*step + window > T): break
            signal_win = signal[:, j*step:j*step + window]
            X = np.array(
                [
                    pra.transform.stft.analysis(x, nfft, nfft // 4).T
                    for x in signal_win
                ]
            )
            X_ts = torch.tensor(X)#, dtype=torch.complex32)
            if self.device is not None:
                X_ts = X_ts.to(self.device)
            
            absX = torch.abs(X_ts)
            absX[absX < tol] = tol
            pX = X_ts / absX

            CC = []
            frame_num = pX.shape[2]

            for k in self.freq_bins:
                CC.append(torch.mm(pX[:, k, :], torch.conj(pX[:, k, :]).T) / frame_num)
            CC = torch.stack(CC)
           

            CC_flat = CC.reshape((-1, CC.shape[-2] * CC.shape[-2]))[:, mask_triu]            
            
            result = torch.real(CC_flat) * self.mode_mat_flat_real - torch.imag(CC_flat) * self.mode_mat_flat_imag
            result = torch.sum(result, (1,2))/self.freq_bins.shape[0]/CC_flat.shape[1]
            self.SRP_map = torch.maximum(self.SRP_map, result)
 
        self.MAX_POWER = torch.amax(self.SRP_map).item()
        self.Min_POWER = torch.amin(self.SRP_map).item() 
        self.fill_powermap_torch()

    def MUSIC_Map_WINDOW(self, signal, window= 36000, tol = 1e-8): #24000
        n_mic = signal.shape[0]
        assert self.mic_pos.shape[0] == n_mic
        nfft = self.n_fft
        
        step = window
        T = signal.shape[1]
        frame_number = T//step

        MUSIC_node = MUSIC(self.freq_bins, self.mode_vec)

        num_run = 0
        for j in range(0, frame_number):

            if (j*step + window > T): break
            signal_win = signal[:, j*step:j*step + window]

            X = np.array(
                [
                    pra.transform.stft.analysis(x, nfft, nfft // 4).T
                    for x in signal_win
                ]
            )

            self.SRP_map += MUSIC_node.MUSIC_process(X)
            num_run += 1

        self.SRP_map = self.SRP_map/num_run

        self.MAX_POWER = np.amax(self.SRP_map) 
        self.Min_POWER = np.amin(self.SRP_map) 
        self.fill_powermap()


    def TOPS_Map_WINDOW(self, signal, window= 36000, tol = 1e-8): #24000
        n_mic = signal.shape[0]
        assert self.mic_pos.shape[0] == n_mic
        nfft = self.n_fft
        window = 72000
        step = window
        T = signal.shape[1]
        frame_number = T//step 
        
        TOPS_node = TOPS(self.mic_pos,  self.grids, self.freq_bins, self.mode_vec, nfft = self.n_fft, c = self.C, fs = self.FS)
        num_run = 0
        for j in range(0, frame_number):
            if (j*step + window > T): break
            signal_win = signal[:, j*step:j*step + window]

            X = np.array(
                [
                    pra.transform.stft.analysis(x, nfft, nfft // 4).T
                    for x in signal_win
                ]
            )
            self.SRP_map += TOPS_node._process(X)
            num_run += 1

        self.SRP_map = self.SRP_map/num_run
        self.MAX_POWER = np.amax(self.SRP_map) 
        self.Min_POWER = np.amin(self.SRP_map) 
        self.fill_powermap()


    def find_valid_peak_new(self, rato = 4):
        threshold = self.threshold[0]*self.MAX_POWER # self.threshold
        if threshold < self.threshold[1]: threshold = self.threshold[1]
        elif threshold > self.threshold[2]: threshold = self.threshold[2]
        threshold2 = threshold * rato
        print("Adaptive threshold: ", self.MAX_POWER,  threshold, threshold2)
        peaks_index = []

        N_grids = self.grids.shape[0]
        grid_visited = np.zeros((N_grids, ))
        thrds = threshold * (0.9 + 1/self.dis_matrix)
        thrds_stack = [thrds[2:-2, 2:-2] for i in range(self.POWER_MAP.shape[-1]-2)]
        thrds = np.stack(thrds_stack, axis = -1)

        thrds2 = threshold2 * (1 + 1/self.dis_matrix)
        thrds_stack2 = [thrds2[2:-2, 2:-2] for i in range(self.POWER_MAP.shape[-1]-2)]
        thrds2 = np.stack(thrds_stack2, axis = -1)

        maxima = np.zeros_like(self.POWER_MAP[2:-2, 2:-2, 1:-1], dtype=bool)
        NX,NY,NZ = self.POWER_MAP.shape
        bool_matrix = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-1, 1):
                    if dx == 0 and dy ==0 and dz == 0: continue
                    a = self.POWER_MAP[2:-2, 2:-2, 1:-1] >= self.POWER_MAP[2+dx:NX-2+dx, 2+dy:NY-2+dy, 1+dz:NZ-1+dz]
                    bool_matrix.append(a)
  
        bool_matrix.append(self.POWER_MAP[2:-2, 2:-2, 1:-1] > thrds)
        bool_matrix.append(self.POWER_MAP[2:-2, 2:-2, 1:-1] <= thrds2)
        condition2 = np.logical_and.reduce(
            bool_matrix
        )
        
        condition1 = self.POWER_MAP[2:-2, 2:-2, 1:-1] > thrds2

        maxima = np.logical_or(condition2, condition1)
        index_2d =  np.transpose(np.nonzero(maxima))

        for i in range(index_2d.shape[0]):
            _id = self.POWER_INDEX[index_2d[i][0] + 2, index_2d[i][1] + 2, index_2d[i][2] + 1]
            if grid_visited[_id]: continue
            peaks_index.append(_id)
            grid_visited[_id] = 1
        return peaks_index
  

    def local_source_adaptive(self):
        t0 = time.time()
        peak_index = self.find_valid_peak_new()
        print("peak_index: ", len(peak_index))
        if self.device is not None:
            SRP_MAP_CPU = self.SRP_map.cpu().numpy()
        else:
            SRP_MAP_CPU = self.SRP_map.numpy()
        peaks = SRP_MAP_CPU[peak_index]
        peaks_pos = self.grids[peak_index]
        self.peaks = peaks
        self.peaks_pos = peaks_pos
        peaks_sample = np.array([ self.clusters[i].sample_offset for i in peak_index])
        peaks_ids = np.argsort(-1*peaks)
        visited = np.zeros_like(peaks)


        peak_candidate = []
        patch_candidate = []
        num_pair = self.num_mic - 1

        ### interate all SPR peak and then cluster the nearby SRP peak to the highest peak
        for _id in peaks_ids:
            candidate = peaks_pos[_id, :]
            if visited[_id] >= 1: 
                continue
            sample_offsets = peaks_sample[_id]
            peak_candidate.append(candidate)
   
            begin_width = self.WIDTH 
            occupy = np.ones((num_pair,  begin_width))
            strict_bound = 0 #2

            for p in patch_candidate:
                another_offset = p.sample_offset
                another_width = p.width_list
                delta_offsets = another_offset - sample_offsets
                range_low = -begin_width/2
                range_high = begin_width/2
                range_low1 =  delta_offsets - another_width/2 + strict_bound
                range_high1 =  delta_offsets + another_width/2 - strict_bound

                delta1 = int(round((range_low1 - range_high).max()))
                delta2 = int(round((range_high1 - range_low).min()))
                if(delta1 >= 0 or delta2 <= 0):
                    continue
                elif( delta1 < 0 ):
                    if begin_width+delta1 < 0:
                        occupy[:, :] = 0
                    else:
                        occupy[:, begin_width+delta1:] = 0
                elif( delta2 > 0 ):
                    if delta2 > begin_width:
                        occupy[:, :] = 0
                    else:
                        occupy[:, 0:delta2] = 0

            width_list_new = []
            sample_offset_new = []
            all_discard = False

            for i in range(num_pair):
                index_1 = np.where(occupy[i])[0]
                if index_1.shape[0] == 0:
                    all_discard = True
                    break
                else:
                    width_list_new.append(index_1.shape[0])
                    new_offset = int(round(sample_offsets[i] + (index_1[0] + index_1[-1] - begin_width + 1)/2))
                    sample_offset_new.append(new_offset)
            if all_discard:
                continue

            # --------------------------------------
            ## mark the visited grids

            PEAK_INCLUDED = hyperbola_area_sample(peaks_sample, sample_offsets,  begin_width - 2*strict_bound + err_tolerance)
            visited += PEAK_INCLUDED

            width_list_new = np.array(width_list_new)
            sample_offset_new = np.array(sample_offset_new)
            final_point = None
            init_area = hyperbola_area_init(self.Axis_range,  sample_offset_new, width_list_new[0] + err_tolerance, self.Pos_5, self.Offset_5, self.Pos_1, self.Offset_1)

            if init_area is None:
                continue
            if init_area.shape[-1] > 0:
                final_point = init_area
            else:
                continue
            p = Patch(sample_offset_new, width_list_new, final_point, candidate)

            patch_candidate.append(p)

        print("SRP-PHAT candidate number: ", len(patch_candidate))
        self.peak_candidate = np.array(peak_candidate)
        return patch_candidate



    ## visualize the SRP-PHAT MAP and picked hypercubes
    def visualize_each_layer(self, voice_positions, out_dir = None, name = "SRP"):
        extent = self.Axis_range[0][0] , self.Axis_range[0][1], self.Axis_range[1][0] , self.Axis_range[1][1]
        # spk_id = 2
        Range_spk = self.Range_spk
        self.peak_high_prio = np.array(self.peak_high_prio)
        self.peak_low_prio = np.array(self.peak_low_prio)

        
        self.debug_indexs = np.array(self.debug_indexs)

        for i in range(self.Lz):
            i = self.Lz//2
            h = self.z_grids[i]
            fig = plt.figure()
            ax = fig.gca(
                aspect='equal',
                xlim=(Range_spk[0] , Range_spk[1]),
                ylim=(Range_spk[2], Range_spk[3]))
            SRP_map_visual = self.POWER_MAP[:, : , i]
            SRP_map_visual = (SRP_map_visual.T)[::-1, :]
            plt.imshow(SRP_map_visual, vmin=self.Min_POWER, vmax=self.MAX_POWER, extent=extent)
            if  self.peak_high_prio.shape[0] > 0:
                this_layer_peak = self.peak_high_prio[:, 2]
                this_layer_index = np.where(this_layer_peak == i)
                
                x_index = self.peak_high_prio[this_layer_index, 0]
                y_index = self.peak_high_prio[this_layer_index, 1]
                plt.scatter(self.x_grids[x_index], self.y_grids[y_index], c = "white", marker='^')

            if  self.peak_low_prio.shape[0] > 0:
                this_layer_peak = self.peak_low_prio[:, 2]
                this_layer_index = np.where(this_layer_peak == i)
                
                x_index = self.peak_low_prio[this_layer_index, 0]
                y_index = self.peak_low_prio[this_layer_index, 1]
                plt.scatter(self.x_grids[x_index], self.y_grids[y_index], c = "white", marker='*')
            plt.scatter(self.mic_pos[:, 0], self.mic_pos[:, 1], c = "red")
            
            plt.scatter(voice_positions[:, 0], voice_positions[:, 1], c = "black", marker='x')

            plt.title("height = {:.2f}".format(h))
            if out_dir is not None:
                plt.savefig(out_dir + "/" + name + "_h{:.2f}.png".format(h))
            return
        




