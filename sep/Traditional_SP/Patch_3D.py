import numpy as np 

class Patch(object):
    def __init__(self, sample_offset, width_list, area_points, peak_pos = None):
        self.sample_offset = sample_offset
        self.width_list = np.copy(width_list)
        self.area_points = area_points
        #print("Patch:", sample_offset)
        self.num_pair = sample_offset.shape[0]
        self.peak_pos = peak_pos

    def area_size(self):
        if self.area_points is None or self.area_points.shape[1] == 0:
            return 0
        else:
            return self.area_points.shape[1]
    

    def center_pos(self):
        if self.peak_pos is not None:
            return self.peak_pos
        else:
            if self.area_points is None or self.area_points.shape[1] == 0:
                return None
            else:
                return np.mean(self.area_points, axis=1)

    def hyperbola_general_area(self, X, Y, Z, mic_position, sound_speed, fs):
        z = 1
        for i in range(mic_position.shape[0] - 1):
            f1 = lambda x,y : \
                (((X - mic_position[i+1, 0])**2 + (Y - mic_position[i+1, 1])**2 + (Z - mic_position[i+1, 2])**2)**0.5) / sound_speed * fs - \
                (((X - mic_position[0, 0])**2 + (Y - mic_position[0, 1])**2  + (Z - mic_position[0, 2])**2 )**0.5) / sound_speed * fs

            bound1 = [self.sample_offset[i] - self.width_list[i]/2 - 1e-3, self.sample_offset[i] + self.width_list[i]/2 + 1e-3]
            z = z & (f1(X, Y) >= bound1[0]) & (f1(X, Y) <= bound1[1])

        return z.astype(int) 

    def hyperbola_sample(self, offset):
        z = 1
        for i in range(offset.shape[0]):
            _off = offset[i, :]
            bound1 = [self.sample_offset[i] - self.width_list[i]/2 - 1e-3, self.sample_offset[i] + self.width_list[i]/2 + 1e-3]
            z = z & (_off >= bound1[0]) & (_off <= bound1[1])

        return z.astype(int) 


    def check_gt(self, sample_offsets_gt):   
        speaker_num = sample_offsets_gt.shape[1]

        #print(sample_offsets_gt.T)
        #print(self.sample_offset)

        for i in range(speaker_num):
            valid = True
            for j in range(self.num_pair):
                delta = sample_offsets_gt[j, i]
                if abs(delta - self.sample_offset[j]) > self.width_list[j]/2 + 1:
                    valid = False
                    break
            if valid == True:
                return True
        
        return False


    def check_out(self, upper_bound_pairwise):
        for i in range(self.num_pair):
            upper_bound = upper_bound_pairwise[i]
            
            while True:
                
                if abs(self.sample_offset[i]) <= upper_bound or self.width_list[i] <= 4:
                    break
                #print(i, self.sample_offset[i], upper_bound)
                resolution = self.width_list[i]

                if self.sample_offset[i] > upper_bound:
                    self.sample_offset[i] =  self.sample_offset[i] - resolution/4
                elif self.sample_offset[i] < -upper_bound:             
                    self.sample_offset[i] =  self.sample_offset[i] + resolution/4

                self.width_list[i] = resolution/2   
                #print(self.width_list[i])
                #print(self.sample_offset[i])

    def check_ready_Spotforming(self, MIN_TOLERANCE):
        for i in range(self.num_pair):  
            if self.width_list[i] > MIN_TOLERANCE:
                return False, i
        return True, -1