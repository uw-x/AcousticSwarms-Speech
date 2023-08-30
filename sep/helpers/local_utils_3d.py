import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from .constants import SPEED_OF_SOUND, FS, MIN_AREA, MIN_WIDTH, MIN_TOLERANCE, LOC_MODEL_THRESHOLD, MAX_NUM, SPOT_POWER_THRESHOLD1, MAX_BIG_PATCH, USE_RELATIVE_SPOT_POWER, MIN_WIDTH_REQUIRED
from .utils import write_audio_file
from sep.Traditional_SP.Patch_3D import Patch
import time 

from scipy.ndimage import uniform_filter1d


    
def max_avg_power(x: np.ndarray, window_size: int = 12000):
    max_avg_energy = uniform_filter1d(x**2, size=window_size, mode='constant', origin=-window_size//2)	
    max_avg_energy = np.sqrt(np.abs(max_avg_energy))	
    y = np.argmax(max_avg_energy)	
    return max_avg_energy.max(), np.pad(x, (0, window_size))[y:y+window_size]


def visualize_small_patch(mic_positions, voice_positions, Range_spk, center_list, power_list):
    xx = np.arange(-5, 5, 0.01)
    yy = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(xx, yy)

    #print(next_patch)
    fig, ax = plt.subplots()
    ax.set(xlim=(Range_spk[0], Range_spk[1]), ylim = (Range_spk[2], Range_spk[3]))
    ax.set_aspect("equal")

    plt.tick_params(axis='both',
        which='both', bottom='off',
        top='off', labelbottom='off', right='off', left='off', labelleft='off'
    )
    #cmap = matplotlib.colors.ListedColormap(["b","b"])
    power_list = np.array(power_list)
    power_list = (power_list - power_list.min())/(power_list.max() - power_list.min()) + 1e-1
    center_list = np.array(center_list)
    # print(center_list.shape)
    # print(power_list.shape)
    plt.scatter(center_list[:, 0], center_list[:, 1], s = 10, c =power_list, alpha = 1,  vmin = -1, vmax = 1, edgecolors = 'None', cmap = "RdYlGn")

    for idx, voice_position in enumerate(voice_positions):
        voice_position = tuple(voice_position)
        plt.scatter(voice_position[0], voice_position[1], marker='x')
        #ax.add_patch(circle)
        plt.text(x = voice_position[0] + 0.1, y = voice_position[1] + 0.1, s =str(idx))

    for idx, mic_position in enumerate(mic_positions):
        mic_position = tuple(mic_position)
        circle = plt.Circle(mic_position, radius=0.013, color='blue')
        ax.add_patch(circle)
        plt.text(x = mic_position[0] + 0.1, y = mic_position[1] + 0.1, s =str(idx))
    #print("samples: ", samples)

    # ax.set_xlim([-5,5])
    # ax.set_ylim([-5,5])
    ax.tick_params(axis='both', which='both', labelcolor="white", colors="white")


def visualize_result(mic_positions, voice_positions, pair_list, simple_pos = None, Range_spk = None):
    fig, ax = plt.subplots()

    if Range_spk is None:
        min_pos = np.amin(voice_positions, axis=0) - 1
        max_pos = np.amax(voice_positions, axis=0) + 1

        ax.set(xlim=(min_pos[0], max_pos[0]), ylim = (min_pos[1], max_pos[1]))
    else:
        ax.set(xlim=(Range_spk[0], Range_spk[1]), ylim = (Range_spk[2], Range_spk[3]))
    # ax.set(xlim=(0, 8), ylim = (0, 8))
    ax.set_aspect("equal")

    plt.tick_params(axis='both',
        which='both', bottom='off',
        top='off', labelbottom='off', right='off', left='off', labelleft='off'
    )
    
    for idx, voice_position in enumerate(voice_positions):
        voice_position = tuple(voice_position[:2])
        circle = plt.Circle(voice_position, radius=0.2, color='red')
        ax.add_patch(circle)
        plt.text(x = voice_position[0] + 0.1, y = voice_position[1] + 0.1, s =str(idx))

    for idx, mic_position in enumerate(mic_positions):
        mic_position = tuple(mic_position[:2])
        circle = plt.Circle(mic_position, radius=0.013, color='blue')
        ax.add_patch(circle)
        plt.text(x = mic_position[0] + 0.1, y = mic_position[1] + 0.1, s =str(idx))
        
    #print("samples: ", samples)
    for idx, pair in enumerate(pair_list):
        p = pair[0]
        #print(p.center_pos())
        pred_position = tuple(p.center_pos()[:2])
        circle = plt.Circle(pred_position, radius=0.2, color='green')
        ax.add_patch(circle)
        plt.text(x = pred_position[0] - 0.1, y = pred_position[1] - 0.1, s =str(idx))
        #plt.text(x = mic_position[0] + 0.1, y = mic_position[1] + 0.1, s =str(idx))
    
    if simple_pos is not None and len(simple_pos) > 0:
        plt.scatter(simple_pos[:, 0], simple_pos[:, 1], c = 'black', marker='x')
    # ax.set_xlim([-5,5])
    # ax.set_ylim([-5,5])
    ax.tick_params(axis='both', which='both', labelcolor="white", colors="white")


def visualize_single(patch, mic_positions, voice_positions, Range_spk):

    fig, ax = plt.subplots()
    ax.set(xlim=(Range_spk[0]-1, Range_spk[1]+1), ylim = (Range_spk[2]-1, Range_spk[3]+1))
    ax.set_aspect("equal")

    plt.tick_params(axis='both',
        which='both', bottom='off',
        top='off', labelbottom='off', right='off', left='off', labelleft='off'
    )
    points = patch.area_points
    X = points[0,:]
    Y = points[1,:]
    plt.scatter(X, Y, s = 0.1, marker='o', color = 'red')

    for idx, voice_position in enumerate(voice_positions):
        voice_position = tuple(voice_position)
        circle = plt.Circle(voice_position, radius=0.2, color='black')
        ax.add_patch(circle)
        plt.text(x = voice_position[0] + 0.1, y = voice_position[1] + 0.1, s =str(idx))

    for idx, mic_position in enumerate(mic_positions):
        mic_position = tuple(mic_position)
        circle = plt.Circle(mic_position, radius=0.013, color='blue')
        ax.add_patch(circle)
        plt.text(x = mic_position[0] + 0.1, y = mic_position[1] + 0.1, s =str(idx))
    #print("samples: ", samples)
    ax.tick_params(axis='both', which='both', labelcolor="white", colors="white")

def visualize(patch_list, mic_positions, voice_positions, Range_spk):

    fig, ax = plt.subplots()
    ax.set(xlim=(Range_spk[0]-1, Range_spk[1]+1), ylim = (Range_spk[2]-1, Range_spk[3]+1))
    ax.set_aspect("equal")

    plt.tick_params(axis='both',
        which='both', bottom='off',
        top='off', labelbottom='off', right='off', left='off', labelleft='off'
    )

    for patch in patch_list:
        points = patch.area_points
        X = points[0,:]
        Y = points[1,:]
        plt.scatter(X, Y, s = 0.1, alpha = 0.2, marker='o', color = 'red')

    for idx, voice_position in enumerate(voice_positions):
        voice_position = tuple(voice_position)
        circle = plt.Circle(voice_position, radius=0.04, color='yellow')
        ax.add_patch(circle)
        plt.text(x = voice_position[0] + 0.1, y = voice_position[1] + 0.1, s =str(idx))

    for idx, mic_position in enumerate(mic_positions):
        mic_position = tuple(mic_position)
        circle = plt.Circle(mic_position, radius=0.013, color='blue')
        ax.add_patch(circle)
        plt.text(x = mic_position[0] + 0.1, y = mic_position[1] + 0.1, s =str(idx))
    #print("samples: ", samples)
    cmap = matplotlib.colors.ListedColormap(["b","b"])

    ax.tick_params(axis='both', which='both', labelcolor="white", colors="white")



def visualize_color(mic_positions, voice_positions, patch_list, power_list):
    xx = np.arange(-5, 5, 0.01)
    yy = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(xx, yy)

    #print(next_patch)
    fig, ax = plt.subplots()
    ax.set(xlim=(-5, 5), ylim = (-5, 5))
    ax.set_aspect("equal")

    plt.tick_params(axis='both',
        which='both', bottom='off',
        top='off', labelbottom='off', right='off', left='off', labelleft='off'
    )
    #cmap = matplotlib.colors.ListedColormap(["b","b"])
    power_list = np.array(power_list) + 1
    power_list = power_list/power_list.max()
    Z_all = np.zeros_like(X)
    for i, patch in enumerate(patch_list):
        Z = patch.hyperbola_general_area(X, Y, mic_positions, SPEED_OF_SOUND, FS)
        Z = Z*power_list[i]
        Z_all = np.maximum(Z_all, Z)

    plt.pcolormesh(X, Y, Z_all, cmap="Blues", alpha = 1, vmin=0, vmax = 1 )

    for idx, voice_position in enumerate(voice_positions):
        voice_position = tuple(voice_position)
        plt.scatter(voice_position[0], voice_position[1], marker='x')
        #ax.add_patch(circle)
        plt.text(x = voice_position[0] + 0.1, y = voice_position[1] + 0.1, s =str(idx))

    for idx, mic_position in enumerate(mic_positions):
        mic_position = tuple(mic_position)
        circle = plt.Circle(mic_position, radius=0.013, color='blue')
        ax.add_patch(circle)
        plt.text(x = mic_position[0] + 0.1, y = mic_position[1] + 0.1, s =str(idx))
    #print("samples: ", samples)

    ax.tick_params(axis='both', which='both', labelcolor="white", colors="white")


def search_area(patch_list, mic_positions, upper_bound_pairwise):

    finish_patched = []
    finish_samples = []

    points0 =  patch_list[0].area_points
    samples = []

    for i in range(mic_positions.shape[0] - 1):
        _off = (((points0[0, :] - mic_positions[i+1, 0])**2 + (points0[1, :] - mic_positions[i+1, 1])**2 + (points0[2, :] - mic_positions[i+1, 2])**2)**0.5) / SPEED_OF_SOUND * FS - \
            (((points0[0, :] - mic_positions[0, 0])**2 + (points0[1, :] - mic_positions[0, 1])**2  + (points0[2, :] - mic_positions[0, 2])**2 )**0.5) / SPEED_OF_SOUND * FS
        samples.append(_off)
    samples = np.array(samples)
    samples_lists = [samples]

    while True:
        next_patches = []
        next_samples = []
        
        for i, patch in enumerate(patch_list):
            points0 = samples_lists[i]
            #print(patch.sample_offset, patch.width_list)
            if_continue, next_patch, next_sample= binary_area_divide_width(patch, points0, mic_positions, upper_bound_pairwise)
            if if_continue:
                next_patches.extend(next_patch)
                next_samples.extend(next_sample)
            else:
                finish_patched.append(next_patch)
                finish_samples.append(next_sample)

        if (len(next_patches) == 0 ):
            break
        patch_list = next_patches
        samples_lists = next_samples
    return finish_patched

def binary_area_divide_width(patch, samples0,  mic_positions, upper_bound_pairwise):
    if upper_bound_pairwise is not None:
        patch.check_out(upper_bound_pairwise)

    candidates_area = patch.area_points
    candidates = patch.sample_offset
    widths = patch.width_list

    num_points = patch.area_size()#candidates_area.shape[1]
    num_pair = candidates.shape[0]


    if  (np.amax(widths)/2 <= MIN_WIDTH_REQUIRED) and num_points <= MIN_AREA:
        #print(widths)
        return False, patch, samples0

    min_difference = 2500000
    min_patch = None
    min_sample = None
    remain_width_8 = False


    for i in range(num_pair):
        if widths[i]/2 < MIN_WIDTH:
            continue
        two_patches = []
        two_samples = []
        half_candidate0 = np.copy(candidates)
        half_candidate0[i] -= widths[i]/4
        half_candidate1 = np.copy(candidates)
        half_candidate1[i] += widths[i]/4
        
        half_width = np.copy(widths)
        half_width[i] /= 2

        
        patch0 = Patch(half_candidate0, half_width, None)
        patch1 = Patch(half_candidate1, half_width, None)

        area0 = patch0.hyperbola_sample(samples0)
        area0 = (area0 == 1)
        size0 = np.sum(area0)
        if size0 == 0:
            patch0.area_points = None
        else:
            points0 = candidates_area[:, area0]
            patch0.area_points = points0
            two_patches.append(patch0)
            sample0 = samples0[:, area0]
            two_samples.append(sample0)
        area1 = patch1.hyperbola_sample(samples0)
            
        area1 = (area1 == 1)
        size1 = np.sum(area1)

        if size1 == 0:
            patch1.area_points = None
        else:
            points1 = candidates_area[:, area1]
            patch1.area_points = points1
            two_patches.append(patch1)
            sample1 = samples0[:, area1]
            two_samples.append(sample1)


        if half_width[i] > MIN_WIDTH_REQUIRED: #5
            if not remain_width_8:
                min_difference = abs(size0 - size1)
                min_patch = two_patches
                remain_width_8 = True
                min_sample = two_samples
            else:
                if abs(size0 - size1) < min_difference:
                    min_difference = abs(size0 - size1)
                    min_patch = two_patches
                    min_sample = two_samples
        else:
            if not remain_width_8:
                if abs(size0 - size1) < min_difference:
                    min_difference = abs(size0 - size1)
                    min_patch = two_patches
                    min_sample = two_samples

    if min_patch is None or len(two_patches) == 0:
        #print(widths)
        return False, patch, samples0

    return True, min_patch, min_sample



def binary_search_baseline(mix_data, spot_model, patch_list, mic_positions):    
    # t0 = time.time()
    sep_data = spot_model.shift_and_sep(mix_data, patch_list, Strict = 0)
    # t1 = time.time()

    powers = []
    powers_win = []
    powers_with_dis = []


    for i in range(sep_data.shape[0]):
        sep_data[i,:] = sep_data[i,:] - np.mean(sep_data[i,:])
        p0 = np.sum(sep_data[i,:]**2)
        powers.append(p0)
        p, _ = max_avg_power(sep_data[i,:])
        powers_win.append(p)

        if patch_list[i].center_pos().shape[0] == 3:
            d = np.linalg.norm(patch_list[i].center_pos() - mic_positions[0])
        else:
            d = 4
        powers_with_dis.append(  p*(d + 1) )
            # if gt_label[i] == True:
            #     write_audio_file("./debug/" + str(i) + ".wav", sep_data[i,:], sr=48000)
                # write_audio_file("./debug/" + str(i) + "_input.wav", inp[i], sr=48000)
    
    valid_patch = []
    
    sort_idx = np.argsort(-1*np.array(powers_win))
    max_power_with_dis = max(powers_with_dis)

    if USE_RELATIVE_SPOT_POWER:
        relative_threshold = min([0.4*max_power_with_dis, SPOT_POWER_THRESHOLD1])
        # if visual_save:
        #     print( "relative_threshold: " , relative_threshold , 0.4*max_power_with_dis, SPOT_POWER_THRESHOLD1)
    else:
        relative_threshold = SPOT_POWER_THRESHOLD1


    for i in sort_idx:
        if powers_with_dis[i] < relative_threshold :
            continue
        if len(valid_patch) >= MAX_BIG_PATCH:
            print("warning too many patch remaining, only keep the best 30")
            break
        valid_patch.append(patch_list[i])    

    # print("biG patch model infer time = ",  t1 - t0)
    # print("T_model = ", t1 - t0, "T_2 = ", t2 - t1, big_num, (t1 - t0)/big_num)
    return valid_patch, powers_with_dis, relative_threshold*1.2
