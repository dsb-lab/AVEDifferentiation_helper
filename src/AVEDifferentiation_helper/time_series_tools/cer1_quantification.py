
import numpy as np
from scipy.ndimage import gaussian_filter

from .general_tools import compute_distance_xy, divide_line_1d
from scipy.ndimage.filters import gaussian_filter1d


def compute_cer1_nuc_background(t, channel1, channel2, CTB, ntiles=10, sigma=0, tile_stack_to_return=None, return_all_backgrounds=False):

    # Each cell has a different value
    tile_sizes = divide_line_1d(0, CTB.hyperstack.shape[-1], ntiles)
    tile_centers_1D = np.rint(tile_sizes[1:] - np.diff(tile_sizes)/2).astype(int)
    tile_limsx = np.array([[tile_sizes[i],tile_sizes[i+1]] for i in range(ntiles)])
    tile_limsy = np.array([[tile_sizes[i],tile_sizes[i+1]] for i in range(ntiles)])

    tile_lims = np.array([[x,y] for x in tile_limsx for y in tile_limsy])

    tile_centers_2D = np.array([[x,y] for x in tile_centers_1D for y in tile_centers_1D])

    cer1_backgrounds = np.zeros_like(CTB.hyperstack[:,0,channel1])
    nuc_backgrounds = np.zeros_like(CTB.hyperstack[:,0,channel1])
    if tile_stack_to_return is None:
        # Calculate the centroid
        centroid = np.mean(tile_centers_2D, axis=0)

        # Find the closest point to the centroid
        distances = np.linalg.norm(tile_centers_2D - centroid, axis=1)
        tile_stack_to_return = np.argmin(distances)

    cer1_background = np.zeros_like(CTB.hyperstack[0,0,channel1], dtype="float32")
    nuc_background = np.zeros_like(CTB.hyperstack[0,0,channel1], dtype="float32")

    cer1_background_nofilter = np.zeros_like(CTB.hyperstack[0,0,channel1], dtype="float32")
    nuc_background_nofilter = np.zeros_like(CTB.hyperstack[0,0,channel1], dtype="float32")

    if hasattr(t, '__iter__'):
        times = t
    else: 
        times = [t]
    
    if tile_stack_to_return is None:
        tile_stack_to_return = np.rint(len(tile_centers_2D)/2).astype(int)
    
    for time in times:
        print(time)
        stack_nuc = gaussian_filter(CTB.hyperstack[time,0,channel1], sigma)
        stack_cer1 = gaussian_filter(CTB.hyperstack[time,0,channel2], sigma)
    
        _cer1_background = np.zeros_like(stack_nuc)
        _nuc_background = np.zeros_like(stack_nuc)

        labs = []
        tile_cells_id = []
        for cell in CTB.jitcells:
            if time not in cell.times:
                continue
            
            labs.append(cell.label)
            tid = cell.times.index(time)
            center = cell.centers[tid][1:]
            distances = [compute_distance_xy(center[0], tile_center[0], center[1], tile_center[1]) for tile_center in tile_centers_2D]
            tileid = np.argmin(distances)
            tile_cells_id.append(tileid)

        for tile_id in range(len(tile_centers_2D)):
            
            tile_xlims = tile_lims[tile_id][0]
            tile_ylims = tile_lims[tile_id][1]
            
            tile_stack = 2*np.ones_like(stack_nuc)
            tile_stack[tile_ylims[0]:tile_ylims[1], tile_xlims[0]:tile_xlims[1]] = 0
            
            if tile_id == tile_stack_to_return:
                tile_sctak_return = tile_stack
            for lab in labs:
                cell = CTB._get_cell(lab)
                tid = cell.times.index(time)
                mask = cell.masks[tid][0]
                for m in mask:
                    tile_stack[m[1], m[0]] = 1
            # NUC
            ids_back = np.where(tile_stack==0)
            nuc_back = np.mean(stack_nuc[ids_back[0], ids_back[1]])
            
            # CER1
            ids_back = np.where(tile_stack==0)
            c1_back = np.mean(stack_cer1[ids_back[0], ids_back[1]])
            
            _cer1_background[tile_ylims[0]:tile_ylims[1], tile_xlims[0]:tile_xlims[1]] = c1_back
            _nuc_background[tile_ylims[0]:tile_ylims[1], tile_xlims[0]:tile_xlims[1]] = nuc_back
           
        cer1_background_nofilter += _cer1_background
        nuc_background_nofilter  += _nuc_background
         
        _cer1_background = gaussian_filter(_cer1_background, tile_sizes[1])
        _nuc_background  = gaussian_filter(_nuc_background, tile_sizes[1])
        
        cer1_backgrounds[time] = _cer1_background
        nuc_backgrounds[time] = _nuc_background

        cer1_background += _cer1_background
        nuc_background  += _nuc_background
    
    cer1_background = np.divide(cer1_background, np.float(len(times)))
    nuc_background  = np.divide(nuc_background, np.float(len(times)))

    cer1_background_nofilter = np.divide(cer1_background_nofilter, np.float(len(times)))
    nuc_background_nofilter  = np.divide(nuc_background_nofilter, np.float(len(times)))
    if return_all_backgrounds:
        return cer1_background_nofilter, nuc_background_nofilter, cer1_background, nuc_background, tile_sctak_return, np.mean(np.diff(tile_sizes)).astype(int), cer1_backgrounds, nuc_backgrounds
    else:
        return cer1_background_nofilter, nuc_background_nofilter, cer1_background, nuc_background, tile_sctak_return, np.mean(np.diff(tile_sizes)).astype(int)

def quantify_mean_nuc(CTB, nuc_background, max_time=None, sigma=0, norm_back=False):
    if max_time is None:
        max_time=CTB.times
    
    mean_nucs = []
    for t, labels in enumerate(CTB.unique_labels_T[:max_time]):
        if norm_back:
            stack_nuc_wo_background = gaussian_filter(CTB.hyperstack[t,0,0], sigma) - nuc_background[t]
        else:
            stack_nuc_wo_background = gaussian_filter(CTB.hyperstack[t,0,0], sigma)
            
        nucs = []
        for lab in labels:
            cell = CTB._get_cell(lab)
            
            tid = cell.times.index(t)
            mask = cell.masks[tid][0]
                           
            nuc_val = np.mean(stack_nuc_wo_background[mask[:,1], mask[:,0]])               
            nucs.append(nuc_val)

        mean_nucs.append(np.mean(nucs))
    return mean_nucs

def quantify_mean_cer1(CTB, nuc_background, cer1_background, max_time=None, sigma_image=0, sigmat=0.0, norm_back=False, norm_nuc=False):
    if max_time is None:
        max_time=CTB.times
    
    mean_nucs = []
    mean_cer1s = []
    CER1s = {}
    NUCs = {}
    quanfity_cer1_nuc_dict(CER1s, NUCs, range(CTB.times),  0, 1, CTB, cer1_background, nuc_background, sigma_image=sigma_image, sigmat=sigmat, norm_back=norm_back, norm_nuc=norm_nuc)
    for t, labels in enumerate(CTB.unique_labels_T[:max_time]):
        if norm_back:
            stack_nuc_wo_background = gaussian_filter(CTB.hyperstack[t,0,0], sigma_image) - nuc_background[t]
        else:
            stack_nuc_wo_background = gaussian_filter(CTB.hyperstack[t,0,0], sigma_image)
            
        nucs = []
        cer1 = []
        for lab in labels:
            cell = CTB._get_cell(lab)
            
            tid = cell.times.index(t)
            mask = cell.masks[tid][0]
            
            if tid < 2: continue
            if (tid > len(cell.times)-3) and t < CTB.times-3: continue
            cer1_val = CER1s[lab][tid]
            cer1.append(cer1_val)
                        
            nuc_val = np.mean(stack_nuc_wo_background[mask[:,1], mask[:,0]])     
            nucs.append(nuc_val)

        mean_nucs.append(np.mean(nucs))
        mean_cer1s.append(np.mean(cer1))

    return mean_nucs, mean_cer1s


def quanfity_cer1_nuc(CER1, NUC, t, channel1, channel2, CTB, cer1_background, nuc_background, sigma_image=0, sigmat=0, norm_back=False, norm_nuc=False):
    
    if hasattr(t, '__iter__'):
        times = t
    else: 
        times = [t]
    
    if norm_nuc:
        mean_nucs = quantify_mean_nuc(CTB, nuc_background, sigma=sigma_image)
    
    if norm_back:       
        stack_nuc_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel1], sigma_image) - nuc_background[_t] for _t in times]
        stack_cer1_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel2], sigma_image) - cer1_background[_t] for _t in times]
    else:
        stack_nuc_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel1], sigma_image) for _t in times]
        stack_cer1_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel2], sigma_image) for _t in times]

    for cell in CTB.jitcells:
        cer1 = []
        nuc = []
        for tid, _t in enumerate(cell.times):
            if _t not in times: continue
 
            mask = cell.masks[tid][0]
            
            t_stack = times.index(_t)
            
            nuc_val = np.mean(stack_nuc_wo_background[t_stack][mask[:,1], mask[:,0]]) 
            cer1_val = np.mean(stack_cer1_wo_background[t_stack][mask[:,1], mask[:,0]]) 
            
            nuc.append(nuc_val)
            if norm_nuc:
                mean_nuc = np.mean(mean_nucs)
                # mean_nuc = mean_nucs[_t]
                cer1.append(mean_nuc*cer1_val/nuc_val)
            else:
                cer1.append(cer1_val)

        if sigmat==0:
            CER1.append(cer1)
            NUC.append(nuc)
        else:
            CER1.append(gaussian_filter1d(cer1, sigmat))
            NUC.append(gaussian_filter1d(nuc, sigmat))

    return 

def quanfity_cer1_nuc_dict(CER1, NUC, t,  channel1, channel2, CTB, cer1_background, nuc_background, sigma_image=0, sigmat=0, norm_back=False, norm_nuc=False):
    
    if hasattr(t, '__iter__'):
        times = t
    else: 
        times = [t]

    if norm_nuc:
        mean_nucs = quantify_mean_nuc(CTB, nuc_background, sigma=sigma_image)
    
    if norm_back:       
        stack_nuc_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel1], sigma_image) - nuc_background[_t] for _t in times]
        stack_cer1_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel2], sigma_image) - cer1_background[_t] for _t in times]
    else:
        stack_nuc_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel1], sigma_image) for _t in times]
        stack_cer1_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel2], sigma_image) for _t in times]
    
    
    # mins_stack = [np.percentile(CTB.hyperstack[_t,0,channel1], 0.01) for _t in times]
    # stack_nuc_wo_background = [CTB.hyperstack[_t,0,channel1] - mins_stack[_t] for _t in times]

    for cell in CTB.jitcells:
        cer1 = []
        nuc = []
        for tid, _t in enumerate(cell.times):
            
            if _t not in times: continue
    
            mask = cell.masks[tid][0]
            
            t_stack = times.index(_t)
            
            nuc_val = np.mean(stack_nuc_wo_background[t_stack][mask[:,1], mask[:,0]]) 
            cer1_val = np.mean(stack_cer1_wo_background[t_stack][mask[:,1], mask[:,0]]) 

            nuc.append(nuc_val)
            if norm_nuc:
                mean_nuc = np.mean(mean_nucs)
                # mean_nuc = mean_nucs[_t]
                cer1.append(mean_nuc*cer1_val/nuc_val)
            else:
                cer1.append(cer1_val)
        
        if sigmat==0:
            CER1[cell.label] = cer1
            NUC[cell.label] = nuc
        else:    
            CER1[cell.label] = gaussian_filter1d(cer1, sigmat)
            NUC[cell.label] = gaussian_filter1d(nuc, sigmat)
            
    return 

def quanfity_cer1_nuc_per_time(CER1, CTB, max_time, cer1_th):
    cer1_positives = []
    total_cells = []
    for t, labels in enumerate(CTB.unique_labels_T[:max_time]):
        cer1_positives.append(0)
        total_cells.append(0)
        for lab in labels:
            total_cells[t] += 1
            cer1_t = CER1[lab]
            cell = CTB._get_cell(lab)
            tid = cell.times.index(t)
            cer1 = cer1_t[tid]
            if cer1 > cer1_th:
                cer1_positives[t] += 1
                
    return cer1_positives, total_cells


# def quanfity_cer1_nuc_per_time(channel1, channel2, CTB, cer1_background, nuc_background, sigma_image=0, norm_back=False, norm_nuc=False):
    
#     CER1 = []
#     NUC  = []

#     for t in range(CTB.times):
#         CER1.append([])
#         NUC.append([])
    
#     if norm_nuc:
#         mean_nucs = quantify_mean_nuc(CTB, nuc_background, sigma=sigma_image, norm_back=norm_back)
    
#     if norm_back:       
#         stack_nuc_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel1], sigma_image) - nuc_background[_t] for _t in range(CTB.times)]
#         stack_cer1_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel2], sigma_image) - cer1_background[_t] for _t in range(CTB.times)]
#     else:
#         stack_nuc_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel1], sigma_image) for _t in range(CTB.times)]
#         stack_cer1_wo_background = [gaussian_filter(CTB.hyperstack[_t,0,channel2], sigma_image) for _t in range(CTB.times)]

#     for cell in CTB.jitcells:
#         for tid, _t in enumerate(cell.times):
 
#             mask = cell.masks[tid][0]
            
#             nuc_val = np.mean(stack_nuc_wo_background[_t][mask[:,1], mask[:,0]]) 
#             cer1_val = np.mean(stack_cer1_wo_background[_t][mask[:,1], mask[:,0]]) 
            
#             NUC[_t].append(nuc_val)
#             if norm_nuc:
#                 mean_nuc = np.mean(mean_nucs)
#                 # mean_nuc = mean_nucs[_t]
#                 CER1[_t].append(mean_nuc*cer1_val/nuc_val)
#             else:
#                 CER1[_t].append(cer1_val)

#     return CER1, NUC, mean_nucs


